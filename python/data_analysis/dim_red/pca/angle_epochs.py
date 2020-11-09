from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as data 

import plotting as plot 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / np.linalg.norm(vector) 
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 
    
def coefs_clf(X_trials, y_trials, clf=LogisticRegression(), shuffle=0): 

    coefs = [] 
    for bin in np.arange(0, X_trials.shape[2]): # epochs 
        X = X_trials[:,:,bin] 
        X = StandardScaler().fit_transform(X) 
        
        if bin==0: 
            y = y_trials 
            if shuffle: 
                random.shuffle(y) 

        clf.fit(X, y)
        
        coefs.append(clf.coef_.flatten() ) 
        # coefs.append( np.concatenate( (mean_intercept_CV.flatten() , mean_coefs_CV.flatten() ), axis=0) ) 
   
    coefs = np.asarray(coefs) 
    return coefs

def get_cos(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """
    alphas = []
    cos_alp=[]
    for j in np.arange(0, coefs.shape[0]): 
        alpha = angle_between(coefs[0], coefs[j]) 
        alphas.append(alpha) 
        cos_alp.append(np.cos(alpha))
        
    return alphas, cos_alp

def angle_epochs(X_proj, IF_SHUFFLE=0):
    
    clf = LogisticRegression(C=1, solver='liblinear', penalty='l2',tol=1e-6, max_iter=int(1e6)) 
    clf = LinearDiscriminantAnalysis() 
    # clf = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto') 
    gv.epochs = ['ED','MD','LD'] 

    for i, gv.trial in enumerate(gv.trials): 
        X_S1_trials = X_proj[i,0] 
        X_S2_trials = X_proj[i,1] 
        X_trials, y_trials = data.get_X_y_epochs(X_S1_trials, X_S2_trials) 
        print('X', X_trials.shape,'y', y_trials.shape) 
        coefs = coefs_clf(X_trials, y_trials, clf=clf) 

        print('coefs', coefs.shape) 
        alpha, cos_alp = get_cos(coefs) 
        print('trial', gv.trial, 'cos_alp', cos_alp) 

        mean_cos = []
        q1 = []
        q3 = []
        if IF_SHUFFLE: 
            mat_cos = [] 
                
            for i in range(100): 
                coefs_shuffle = coefs_clf(X_trials, y_trials, clf=clf, shuffle=1) 
                alpha_shuffle, cos_alp_shuffle = get_cos(coefs_shuffle) 
                
                mat_cos.append(cos_alp_shuffle) 
                
            mat_cos = np.asarray(mat_cos) 
                
            mean_cos = np.mean(mat_cos, axis=0) 
            std_cos = np.std(mat_cos, axis=0) 
            q1 = np.percentile(mat_cos, 25, axis=0) 
            q3 = np.percentile(mat_cos, 75, axis=0) 
            
            print('<cos(alp)>', mean_cos, 'std_cos', std_cos, 'q1', q1, 'q3', q3) 
            plot.plot_cosine_bars(cos_alp, mean_cos=[], q1=[], q3=[], IF_SHUFFLE=IF_SHUFFLE) 
