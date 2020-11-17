from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 
from scipy.spatial import distance 

import data.constants as gv 
import data.plotting as plot 

from joblib import Parallel, delayed, parallel_backend
import multiprocessing 

def bootstrap_clf_par(X, y, clf, dum): 

    if dum==1: 
        print('no boot') 
        idx_trials = np.arange(0, X.shape[0]) 
    else: 
        idx_trials = np.hstack( ( np.random.randint(0, int(X.shape[0]/2), int(X.shape[0]/2)),
                                  np.random.randint(int(X.shape[0]/2), X.shape[0], int(X.shape[0]/2)) ) ) 

    X_sample = X[idx_trials] 
    y_sample = y[idx_trials] 

    # for trial in idx_trials: 
    #     idx_neurons = np.random.randint(0, X.shape[1], X.shape[1])
    #     X_sample[trial] = X[trial, idx_neurons] 
            
    # X_sample = StandardScaler().fit_transform(X_sample.T).T
    scaler = StandardScaler().fit(X.T) 
    X_sample = scaler.transform(X_sample.T).T 
    
    clf.fit(X_sample, y_sample) 
    coefs_samples = clf.coef_.flatten() 
    
    return coefs_samples 

def unit_vector(vector): 
    """ Returns the unit vector of the vector.  """ 
    u = vector / (np.linalg.norm(vector) + gv.eps)
    return u 

def angle_between(v1, v2): 
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """ 
    v1_u = unit_vector(v1) 
    v2_u = unit_vector(v2) 
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) 

def get_cos(coefs): 
    """ Returns the cosine of the angle alpha between vector coefs[0] (early delay) and coefs[1] (late delay) """ 
    alphas = [] 
    cos_alp=[]
    
    for j in np.arange(0, coefs.shape[0]): 
        alpha = angle_between(coefs[0], coefs[j]) 
        alphas.append(alpha) 
        cos_alp.append(np.cos(alpha)) 
        
    return alphas, cos_alp 

def EDvsLD(X_proj, IF_CONCAT, IF_EDvsLD, C=1e0, penalty='l2', solver = 'liblinear'): 

    gv.n_boot = int(1e3) 
    num_cores = -int(3*multiprocessing.cpu_count()/4) 

    print(gv.n_boot) 
    # clf = LogisticRegression(C=C, solver=solver, penalty=penalty, tol=1e-3, max_iter=int(1e3), fit_intercept=bool(not gv.standardize), n_jobs=num_cores) 
    # clf = svm.LinearSVC(C=C, penalty=penalty, loss='squared_hinge', dual=False, tol=1e-6, max_iter=int(1e8), fit_intercept=bool(not gv.standardize) )
    
    clf = LinearDiscriminantAnalysis(tol=1e-6, solver='lsqr', shrinkage='auto') 
    
    NO_PCA = 1 
    if X_proj.shape[3]!=gv.n_neurons: 
        NO_PCA = 0 
        X_proj = X_proj[:,:,:,0:gv.n_components,:] 
    print(X_proj.shape) 
        
    if IF_CONCAT:
        print('Concatenated Stim and ED')
    if IF_EDvsLD:
        print('angle btw ED and other epochs')
        
    if IF_EDvsLD: 
        gv.epochs = ['ED', 'MD', 'LD'] 
    else: 
        gv.epochs = ['Stim', 'ED', 'MD', 'LD'] 


    if gv.ED_MD_LD :
        X_ED = np.mean(X_proj[:,:,:,:,0:len(gv.bins_ED)],axis=-1) 
        X_MD = np.mean(X_proj[:,:,:,:,len(gv.bins_ED):len(gv.bins_ED)+len(gv.bins_MD)],axis=-1) 
        X_LD = np.mean(X_proj[:,:,:,:,len(gv.bins_ED)+len(gv.bins_MD):len(gv.bins_ED)+len(gv.bins_MD)+len(gv.bins_LD)],axis=-1) 
        X_stim = X_ED 
    else:
        X_ED = np.mean(X_proj[:,:,:,:,gv.bins_ED[-3:]-gv.bin_start],axis=-1) 
        X_MD = np.mean(X_proj[:,:,:,:,gv.bins_MD[-3:]-gv.bin_start],axis=-1) 
        X_LD = np.mean(X_proj[:,:,:,:,gv.bins_LD[-3:]-gv.bin_start],axis=-1) 
    
    if gv.DELAY_ONLY: 
        X_stim = X_ED 
    else: 
        X_stim = np.mean(X_proj[:,:,:,:,gv.bins_STIM[-3:]-gv.bin_start],axis=-1) 
    
    if IF_CONCAT:
        X_stim_S1 = []
        X_stim_S2 = []

        X_ED_S1 = []
        X_ED_S2 = [] 
        for i in range(X_proj.shape[0]): 
            X_stim_S1.append(X_stim[i,0]) 
            X_stim_S2.append(X_stim[i,1]) 

            X_ED_S1.append(X_ED[i,0]) 
            X_ED_S2.append(X_ED[i,1]) 

        X_stim_S1 = np.vstack(np.asarray(X_stim_S1)) 
        X_stim_S2 = np.vstack(np.asarray(X_stim_S2)) 

        X_ED_S1 = np.vstack(np.asarray(X_ED_S1)) 
        X_ED_S2 = np.vstack(np.asarray(X_ED_S2)) 

    cos_samples_trials = [] 
    dist_samples_trials = [] 
    for n_trial, gv.trial in enumerate(gv.trials): 

        if not IF_CONCAT: 
            X_stim_S1 = X_stim[n_trial,0] 
            X_stim_S2 = X_stim[n_trial,1] 

            X_ED_S1 = X_ED[n_trial,0]        
            X_ED_S2 = X_ED[n_trial,1] 

        X_MD_S1 = X_MD[n_trial,0] 
        X_MD_S2 = X_MD[n_trial,1] 
        
        X_LD_S1 = X_LD[n_trial,0] 
        X_LD_S2 = X_LD[n_trial,1] 
        
        coefs = [] 
        X_stim_S1_S2 = np.vstack([X_stim_S1, X_stim_S2]) 
        X_ED_S1_S2 = np.vstack([X_ED_S1, X_ED_S2]) 
        X_MD_S1_S2 = np.vstack([X_MD_S1, X_MD_S2]) 
        X_LD_S1_S2 = np.vstack([X_LD_S1, X_LD_S2]) 
            
        if IF_EDvsLD: 
            X_epochs = [X_ED_S1_S2 , X_MD_S1_S2 , X_LD_S1_S2 ] 
        else: 
            X_epochs = [X_stim_S1_S2, X_ED_S1_S2 , X_MD_S1_S2 , X_LD_S1_S2 ] 

        coefs = np.empty((len(X_epochs), gv.n_boot, X_ED_S1_S2.shape[1])) 
        for n_epochs, X in enumerate(X_epochs): 
            y = np.array([np.zeros(int(X.shape[0]/2)), np.ones(int(X.shape[0]/2))]).flatten() 
            boot_coefs = Parallel(n_jobs=num_cores, verbose=True)(delayed(bootstrap_clf_par)(X, y, clf, gv.n_boot) for i in range(gv.n_boot)) 
            coefs[n_epochs] = np.array(boot_coefs) 
        
        cos_samples = [] 
        alpha_samples = [] 
        for boot_sample in range(coefs.shape[1]): 
            alpha, cos_alp = get_cos(coefs[:,boot_sample,:])  
            alpha_samples.append(alpha) 
            cos_samples.append(cos_alp) 
                
        cos_samples_trials.append(cos_samples)
        cos_samples = np.asarray(cos_samples) 
        alpha_samples = np.asarray(alpha_samples) 
            
        # mean_cos =  np.mean( np.cos(alpha_samples), axis=0)
        
        # _, mean_cos = np.array( get_cos(np.mean(coefs, axis=1) ) ) 
        # _, q1 = np.array( get_cos(np.percentile(coefs, 25, axis=1) ) ) 
        # _, q3 = np.array( get_cos(np.percentile(coefs, 75, axis=1) ) ) 

        # q1 = mean_cos - q1 
        # q3 = q3 - mean_cos 
        
        mean_cos = np.mean(cos_samples, axis=0) 
        q1 = mean_cos - np.percentile(cos_samples, 25, axis=0) 
        q3 = np.percentile(cos_samples, 75, axis=0) - mean_cos 
        
        print('trial', gv.trial, 'cos', mean_cos, 'q1', q1, 'q3', q3) 
        plot.plot_cosine_bars(mean_cos, [], q1, q3) 
        
    cos_samples_trials = np.asarray(cos_samples_trials)

    cols = [-4/10, -1/10, 2/10] 
    high = [1.3, 1.1] 

    p_values = [] 
    for i in range(1, cos_samples_trials.shape[2]): # epochs
        for j in range(1, cos_samples_trials.shape[0]): # trials 
            sample_1  = cos_samples_trials[0,:,i] 
            sample_2  = cos_samples_trials[j,:,i]
            t_score, p_value = stats.ttest_ind(sample_1, sample_2, equal_var=False)
            # if t_score>0:
            #     p_value = p_value/2
            # else:
            #     p_value = 1-p_value/2
            p_values.append(p_value) 
            # print(gv.epochs[i], 'ND vs', gv.trials[j], 't_score', t_score, 'p_value', p_value) 
            
    p_values = np.asarray(p_values).reshape(len(gv.epochs)-1, 2)

    # cols = [-4/10, -1/10, 2/10] 
    # high = [0.8, 0.6] 
    for i in range(0,len(gv.epochs)-1): 
        for j in range(1, len(cols)): 
            plt.plot( [i+cols[0], i+cols[j]] , [high[j-1], high[j-1]] , lw=.8, c='k') 
            if p_values[i,j-1]<=.001: 
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.05, "***", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[i,j-1]<=.01: 
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.05, "**", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[i,j-1]<=.05: 
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1]-.05, "*", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[i,j-1]>.05: 
                plt.text((2*i+cols[0]+cols[j])*.5, high[j-1], "ns", ha='center', va='bottom', color='k', fontsize=6) 

    plt.ylim([-1, 1.5]) 

    if NO_PCA: 
        plot.figDir('') 
    else: 
        plot.figDir('pca') 
        
    if gv.laser_on: 
        gv.figdir = gv.figdir + '/laser_on/' 
        figtitle = '%s_%s_cos_alpha_pca_laser_on' % (gv.mouse, gv.session) 
    else: 
        figtitle = '%s_%s_cos_alpha_pca_laser_off' % (gv.mouse, gv.session) 
    
    gv.figdir = gv.figdir + '/clf/'
    clf_name = clf.__class__.__name__ 
        
    if(clf_name == 'LogisticRegression'): 
        clf_param = '/C_%.3f_penalty_%s_solver_%s/' % (C, penalty, solver) 
        gv.figdir = gv.figdir + clf_name + clf_param 
    else: 
        clf_param = '/C_%.3f/' % C 
        gv.figdir = gv.figdir + clf_name + clf_param 

    if IF_EDvsLD: 
        gv.figdir = gv.figdir + '/EDvsLD/'
        
    if IF_CONCAT:
        gv.figdir = gv.figdir + '/concat_stim_ED/'

    if not os.path.isdir(gv.figdir):
        os.makedirs(gv.figdir)
        
    plot.save_fig(figtitle) 
