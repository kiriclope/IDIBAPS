import numpy as np 

from . import constants as gv 
from . import utils as data 

from scipy.special import erfc 

def synthetic_data(prop_ortho): 
    
    X, y = data.get_fluo_data() 
    print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
    
    # data.get_delays_times() 
    # data.get_bins()
    
    # if gv.mouse in [gv.mice[0]]: 
    #     gv.n_trials = 40 
    # else: 
    #     gv.n_trials = 32 
    
    n_ortho = int(prop_ortho * gv.n_neurons) 
    print('synthetic_data:', 'n_ortho', n_ortho) 
    
    X = np.empty( ( len(gv.trials), 2, int(gv.n_trials/2), len(gv.bins), gv.n_neurons) )
    print('X_syn', X.shape)
    for i, gv.trial in enumerate(gv.trials): 
        X_S1 = np.empty((int(gv.n_trials/2), len(gv.bins), gv.n_neurons)) 
        X_S2 = np.empty((int(gv.n_trials/2), len(gv.bins), gv.n_neurons)) 
        
        for j in range(int(gv.n_trials/2)): 
            for k in range(len(gv.bins)): 
                # X_S1[j,k] = np.random.normal(0, 4, gv.n_neurons) 
                X_S1[j,k] = np.random.normal(0, 4 - 2 * erfc(k/len(gv.bins)), gv.n_neurons) 
                
                # S2 = np.random.normal(1, 4, gv.n_neurons) 
                S2 = np.random.normal(1, 4 - 2 * erfc(k/len(gv.bins)), gv.n_neurons) 
                
                if 'D1' in gv.trial: 
                    if k in gv.bins_LD: 
                        # S2_ortho = np.random.normal(-1, 4, n_ortho) 
                        S2_ortho = np.random.normal(-1, 4 - 2 * erfc(k/len(gv.bins)), n_ortho) 
                        S2[0:n_ortho] = S2_ortho 
                X_S2[j,k] = S2 
                
        X[i,0] = X_S1 
        X[i,1] = X_S2 
    X = np.moveaxis(X,3,4) 
    y = np.array([np.zeros(X.shape[2]), np.ones(X.shape[2])]).flatten() 
    return X, y
