import data.constants as gv 

import numpy as np
from sklearn.decomposition import PCA, SparsePCA 
from sklearn.preprocessing import StandardScaler 

class pca_methods():
    
    def __init__(self, pca_model='pca', pca_method='hybrid', n_components=None, total_explained_variance=0.9, inflection=False, minka_mle=False, ridge_alpha=.01, alpha=1, verbose=True):

        if 'sparse' in pca_model: 
            self.pca_model = SparsePCA 
        else: 
            self.pca_model = PCA 
            
        self.pca_method = pca_method 
        self.total_explained_variance = total_explained_variance
        
        self.alpha = alpha 
        self.ridge_alpha = ridge_alpha 
        
        self.scaler = StandardScaler(with_mean=True, with_std=False) 
        self.inflection = inflection 
        self.minka_mle = minka_mle
        
        self.n_components = n_components 
        self.list_n_components = np.empty(len(gv.trials))
        self.explained_variance = None 
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
            if (1 - S_ii/float(S_nn)) <= 1 - self.total_explained_variance: 
                return num_components 
            
        return np.maximum(s.shape[0], 1) 

    def sparse_get_optimal_n_components(self, X_pca): 
        R = np.linalg.qr(X_pca.T)[1] 
        
        variance=[]
        for i in range(0, R.shape[0]):
            variance.append(np.square(R[i][i])) 
        R_nn = sum(variance) 
        
        temp_variance=[] 
        for i in range(0, R.shape[0]): 
            temp_variance.append(np.square(R[i][i])) 
            R_ii = sum(temp_variance) 
            if (1 - R_ii/float(R_nn)) <= 1 - self.total_explained_variance: 
                return i, np.array(temp_variance)/float(R_nn) 
            
        return np.maximum(R.shape[0], 1) 
    
    def trial_hybrid(self, X_trials, FIT=1):

        # concatenate average over trials
        print(X_trials.shape) 
        print(len(gv.trials), X_trials.shape[-2], len(gv.samples) * X_trials.shape[-1] )        
        X_avg = np.empty( (len(gv.trials), X_trials.shape[-2], len(gv.samples) * X_trials.shape[-1] ) ) 
            
        for n_trial in range(len(gv.trials)): 
            X_S1 = X_trials[n_trial,0] 
            X_S2 = X_trials[n_trial,1] 
            X_avg[n_trial] = np.hstack( ( np.mean(X_S1, axis=0), np.mean(X_S2, axis=0) ) ) 
            
        # X_avg = np.hstack(X_avg) 
        if self.verbose :
            print('X_avg', X_avg.shape) 

        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), X_trials.shape[-2], X_trials.shape[-1]) )
            
        for n_trial in range(X_trials.shape[0]): 
            # standardize neurons/features across trials/samples
            if FIT==1:
                self.scaler.fit(X_avg[n_trial].T) 
            X_avg[n_trial] = self.scaler.transform(X_avg[n_trial].T).T 
        
            # PCA the trial averaged data
            if FIT==1:
            
                if self.minka_mle:
                    n_components = 'mle'
                    self.pca = self.pca_model(n_components=n_components) 
                else:
                    if self.pca_model==PCA :
                        self.n_components = self.get_optimal_number_of_components(X_avg[n_trial].T)
                        self.pca = self.pca_model(n_components=self.n_components) 
                    else:
                        self.n_components = None
                        self.pca = self.pca_model(n_components=self.n_components, ridge_alpha=self.ridge_alpha, alpha=self.alpha) 
                    
                self.pca.fit(X_avg[n_trial].T) 
                self.n_components = self.pca.n_components_
            
                if self.pca_model==PCA :
                    self.explained_variance = self.pca.explained_variance_ratio_                 
            
                if self.inflection:
                    self.n_components = self.get_inflection_point(self.explained_variance) 
                    self.pca = self.pca_model(n_components=self.n_components) 
                    self.pca.fit(X_avg[n_trial].T) 
                    self.n_components = self.pca.n_components_
                    self.explained_variance = self.pca.explained_variance_ratio_                 
                else:
                
                    if self.pca_model==PCA:
                        self.explained_variance = self.pca.explained_variance_ratio_ 
                
                    if self.pca_model==SparsePCA:
                        self.n_components, self.explained_variance = self.sparse_get_optimal_n_components(X_avg.T) 
                
                    self.list_n_components[n_trial] = self.n_components
            
            if self.verbose : 
                print('trial', gv.trials[n_trial], 'n_pc', self.n_components,
                      'explained_variance', self.explained_variance[0:3], 'total' , np.cumsum(self.explained_variance)[-1]) 
                    
            # X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), n_components, X_trials.shape[-1]) ) 
            for j in range(X_trials.shape[1]): # sample 
                for k in range(X_trials.shape[2]): # trial 
                    trial = self.scaler.transform(X_trials[n_trial,j,k,:,:].T).T # neurons x time = features x samples 
                    X_proj[n_trial,j,k,0:self.n_components] = self.pca.transform(trial.T).T 
                
        if self.verbose: 
            print('X_proj', X_proj.shape)
            
        return X_proj 
    
    def trial_concatenated(self, X_trials, FIT=1):
        
        X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), X_trials.shape[-2] , X_trials.shape[-1]) ) 
        # trials = []
        # For each condition (ND, D1, D2), concatenate individual trials for each sample 
        for n_trial in range(len(gv.trials)) : 
            X_S1_S2 = np.hstack( (np.hstack(X_trials[n_trial,0]), np.hstack(X_trials[n_trial,1])) ) # N_neurons x (N_trials * N_times)
            # trials.append(X_S1_S2) 
            
            # X_concat = np.hstack(trials) # N_neurons x (N_conditions * N_trials * N_times) 
            X_concat = X_S1_S2 # N_neurons x (N_trials * N_times) 
            
            # center neurons/features across trials/samples 
            if FIT==1:
                self.scaler.fit(X_concat.T) 
            X_concat = self.scaler.transform(X_concat.T).T 
         
            if FIT==1:
                if self.minka_mle: 
                    n_components = 'mle' 
                    self.pca = self.pca_model(n_components=n_components) 
                else: 
                    if self.pca_model==PCA :
                        self.n_components = self.get_optimal_number_of_components(X_concat.T) 
                        self.pca = self.pca_model(n_components=self.n_components) 
                    else:
                        self.n_components = None
                        self.pca = self.pca_model(n_components=self.n_components, ridge_alpha=self.ridge_alpha, alpha=self.alpha) 
                    
                # pca on (N_samples x N_features), ie X is N_trials x N_neurons 
                self.pca = self.pca.fit(X_concat.T) 
                self.n_components = self.pca.n_components_
                
                if self.pca_model==PCA :
                    self.explained_variance = self.pca.explained_variance_ratio_ 
            
                if self.inflection:
                    self.n_components = self.get_inflection_point(self.explained_variance) 
                    self.pca = self.pca_model(n_components=self.n_components) 
                    X_concat = self.pca.fit_transform(X_concat.T).T                 
                    self.n_components = self.pca.n_components_ 
                    self.explained_variance = self.pca.explained_variance_ratio_
                else:
                    X_concat = self.pca.transform(X_concat.T).T 
                
                if self.pca_model==PCA:
                    self.explained_variance = self.pca.explained_variance_ratio_
                    
                if self.pca_model==SparsePCA:
                    self.n_components, self.explained_variance = self.sparse_get_optimal_n_components(X_concat.T)
                    
                self.list_n_components[n_trial] = self.n_components 

            else:
                X_concat = self.pca.transform(X_concat.T).T
                
            if self.verbose :
                print('n_pc', self.n_components,'explained_variance', self.explained_variance[0:3],
                      'total' , np.cumsum(self.explained_variance)[-1]*100, 'X_concat', X_concat.shape) 
                
            # X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), n_components , X_trials.shape[-1]) ) 
            # for i in range( len(gv.trials) ): 
            for j in range( len(gv.samples) ) : 
                for k in range( int( gv.n_trials/len(gv.samples) ) ) : 
                    for l in range(self.n_components) : 
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
            pca = self.pca_model(n_components=n_components) 
        else:
            if self.pca_model==PCA :
                self.n_components = self.get_optimal_number_of_components(X_avg.T) 
                pca = self.pca_model(n_components=self.n_components) 
            else:
                self.n_components = None
                pca = self.pca_model(n_components=self.n_components, ridge_alpha=self.ridge_alpha, alpha=self.alpha) 
                
        # pca = self.pca_model(n_components=n_components) 
        pca.fit(X_avg.T)        
        self.n_components = pca.n_components_ 
            
        if self.inflection:
            self.n_components = self.get_inflection_point(self.explained_variance) 
            pca = self.pca_model(n_components=self.n_components) 
            X_proj = pca.fit_transform(X_avg.T).T 
            
            self.n_components = pca.n_components_ 
            self.explained_variance = pca.explained_variance_ratio_ 
        else:
            X_proj = pca.transform(X_avg.T).T
            
            if self.pca_model==PCA :
                self.explained_variance = pca.explained_variance_ratio_
                
            if self.pca_model==SparsePCA:
                self.n_components, self.explained_variance = self.sparse_get_optimal_n_components(X_concat.T)

                
        if self.verbose :
            print('n_pc', self.n_components,'explained_variance', self.explained_variance,
                  'total' , np.cumsum(self.explained_variance)[-1], 'X_proj', X_proj.shape) 
        
        X_proj = np.asarray(X_proj) 
        X_proj = X_proj.reshape(self.n_components, len(gv.trials), len(gv.samples), X_trials.shape[-1]) 
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
    
    def transform(self, X_trials, y=None): 
        X_proj = [] 
        if self.pca_method in 'hybrid': 
            X_proj = self.trial_hybrid(X_trials, FIT=0) 
        if self.pca_method in 'concatenated': 
            X_proj = self.trial_concatenated(X_trials, FIT=0) 
        return X_proj
