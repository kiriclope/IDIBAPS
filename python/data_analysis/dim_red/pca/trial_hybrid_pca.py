from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 

from scipy.signal import savgol_filter

import data.constants as gv 
importlib.reload(gv) 

import data.utils as data 
importlib.reload(data) 

import data.fct_facilities as fac 
importlib.reload(fac) 
fac.SetPlotParams() 

import data.preprocessing as pp 
import data.plotting as pl 
import data.synthetic as syn 

import detrend 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
pc_shift = 0 

gv.explained_variance = 0.90 

gv.scriptdir = os.path.dirname(__file__) 

gv.IF_SAVE=1 
gv.IF_PCA=1 

gv.T_WINDOW = 0

IF_DECONVOLVE = 0 

gv.DELAY_ONLY = 0 
gv.ED_MD_LD = 0
gv.EDvsLD = 1

gv.DOWN_SAMPLING = 0 
gv.bootstrap_trials=0  

gv.standardize = 1 
SYN=0 
gv.detrend = 0 
POLY_DEG = 7 

gv.correct_trial = 0  # 17-14-16 / 6 
gv.laser_on = 0 
gv.pca_concat = 0 

gv.n_components = 15 #'mle' #75% => 11 C57/ChR - 18 Jaws # inflexion 2-4 
gv.data_type= 'fluo' # 90% var 25-42-110 # 95% 43 # 75% 11 

def get_optimal_number_of_components(X): 
    cov = np.dot(X,X.transpose())/float(X.shape[0]) 
    U,s,v = np.linalg.svd(cov) 

    S_nn = sum(s) 

    for num_components in range(0,s.shape[0]):
        temp_s = s[0:num_components]
        S_ii = sum(temp_s)
        if (1 - S_ii/float(S_nn)) <= 1-gv.explained_variance: 
            return num_components

    return s.shape[0] 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[4]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_bins(gv.T_WINDOW) # .9 or 1 
        
        if gv.mouse in [gv.mice[0]]: 
            gv.n_trials = 40 
        else: 
            gv.n_trials = 32 

        if SYN:
            X_syn = syn.synthetic_data(0.5) 

        if gv.DELAY_ONLY: 
            gv.bin_start = gv.bins_delay[0] 
            gv.t_start = gv.t_ED[0] 
            gv.trial_size = len(gv.bins_delay) 
            gv.time = gv.t_delay
        elif gv.ED_MD_LD:
            gv.bin_start = gv.bins_delay[0] 
            gv.t_start = gv.t_ED[0] 
            gv.trial_size = len(gv.bins_ED_MD_LD) 
            gv.time = gv.t_ED_MD_LD 
            
        X_trials = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_neurons, gv.trial_size) ) 
        X_avg = np.empty( (len(gv.trials), gv.n_neurons, len(gv.samples) * gv.trial_size ) ) 

        for n_trial, gv.trial in enumerate(gv.trials):

            if SYN:
                print('synthetic_data') 
                X_S1 = X_syn[n_trial,0] 
                X_S2 = X_syn[n_trial,1] 
            else: 
                X_S1, X_S2 = data.get_S1_S2_trials(X, y) 

            data.get_trial_types(X_S1) 
            
            # compute DF over F0
            if not IF_DECONVOLVE:
                X_S1 = pp.dFF0(X_S1) 
                X_S2 = pp.dFF0(X_S2) 
            else:
                # X_S1 = pp.deconvolve_X(X_S1) 
                # X_S2 = pp.deconvolve_X(X_S2) 
                for trial in range(X_S1.shape[0]): 
                    X_S1[trial] = pp.deconvolve_X(X_S1[trial]) 
                    X_S2[trial] = pp.deconvolve_X(X_S2[trial]) 
                
            if gv.DOWN_SAMPLING: 
                X_S1 = pp.downsample(X_S1,1,2) 
                X_S2 = pp.downsample(X_S2,1,2) 
                gv.trial_size = X_S1.shape[2] 
                
            if gv.DELAY_ONLY: 
                X_S1 = X_S1[:,:, gv.bins_delay] 
                X_S2 = X_S2[:,:, gv.bins_delay] 
            elif gv.ED_MD_LD: 
                X_S1 = X_S1[:,:, gv.bins_ED_MD_LD] 
                X_S2 = X_S2[:,:, gv.bins_ED_MD_LD] 
                
            if gv.detrend: 
                X_trend = [] 
                for trial in range(X_S1.shape[0]): 
                    fit_values = detrend.detrend_data(X_S1[trial], poly_fit=1, degree=POLY_DEG) 
                    X_trend.append(fit_values) 
                X_trend = np.asarray(X_trend) 
                
                X_S1 = X_S1 - X_trend 
                # X_S1 = X_S1 - X_trend[:,np.newaxis,:] 
                
                X_trend = [] 
                for trial in range(X_S2.shape[0]): 
                    fit_values = detrend.detrend_data(X_S2[trial], poly_fit=1, degree=POLY_DEG) 
                    X_trend.append(fit_values)  
                X_trend = np.asarray(X_trend) 
                
                X_S2 = X_S2 - X_trend 
                # X_S2 = X_S2 - X_trend[:,np.newaxis,:]
                
            # for trial in range(X_S2.shape[0]): 
            #     X_S1[trial] = savgol_filter(X_S1[trial], 17, polyorder = 7, deriv=0) 
            #     X_S2[trial] = savgol_filter(X_S2[trial], 17, polyorder = 7, deriv=0) 
            
                # X_S1[trial] = savgol_filter(X_S1[trial], 17, polyorder=2, deriv=2) 
                # X_S2[trial] = savgol_filter(X_S2[trial], 17, polyorder=2, deriv=2) 
            
            X_trials[n_trial,0] = X_S1
            X_trials[n_trial,1] = X_S2
            
            X_avg[n_trial] = np.hstack( ( np.mean(X_S1, axis=0), np.mean(X_S2, axis=0) ) ) 
            
        print('X_trials', X_trials.shape) 
        
        X_avg = np.hstack(X_avg)
        print('X_avg', X_avg.shape) 

        # standardize the trial averaged data 
        scaler = StandardScaler(with_mean=True, with_std=True) 

        # standardize neurons/features across trials/samples
        scaler.fit(X_avg.T) 
        X_avg = scaler.transform(X_avg.T).T 

        # sparse PCA
        # pca = SparsePCA(n_components=gv.n_components) 
        # X_avg = pca.fit_transform(X_avg.T).T

        # PCA the trial averaged data
        gv.n_components = get_optimal_number_of_components(X_avg) 
        pca = PCA(n_components=gv.n_components) 
        X_avg = pca.fit_transform(X_avg.T).T
        
        # X_avg = X_avg.reshape(gv.n_components, len(gv.trials), len(gv.samples), gv.trial_size)
        # X_avg = np.moveaxis(X_avg, 0, 2) 
        
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]) 

        X_proj = np.empty( (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_components, gv.trial_size) ) 
        for i in range(X_trials.shape[0]): 
            for j in range(X_trials.shape[1]): 
                for k in range(X_trials.shape[2]): 
                    trial = scaler.transform(X_trials[i,j,k,:,:].T).T # neurons x time = features x samples 
                    X_proj[i,j,k] = pca.transform(trial.T).T 
                
        print('X_proj', X_proj.shape) 

        # figname = '%s_%s_pca_scree_plot' % (gv.mouse, gv.session) 
        # plt.figure(figname) 
        # plt.plot(explained_variance,'-o') 
        # plt.xlabel('components')
        # plt.ylabel('explained variance') 
        # figdir = pl.figDir() 
        # pl.save_fig(figname)         
        
        # if gv.laser_on:
        #     figname = '%s_%s_pca_laser_on_%d' % (gv.mouse, gv.session, pc_shift)
        # else:
        #     figname = '%s_%s_pca_laser_off_%d' % (gv.mouse, gv.session, pc_shift)

        # plt.figure(figname, figsize=[10, 2.8])    
        # x = gv.time 
        # for n_pc in range(np.amin([gv.n_components,3])): 
        #     ax = plt.figure(figname).add_subplot(1, 3, n_pc+1) 
        #     for i, trial in enumerate(gv.trials): 
        #         for j, sample in enumerate(gv.samples): 
        #             dum = X_proj[i,j,:,n_pc+pc_shift,:].transpose() 
        #             y = np.mean( dum, axis=1) 
        #             y = gaussian_filter1d(y, sigma=1) 
                    
        #             ax.plot(x, y, color=pal[i]) 
        #             ci = pp.conf_inter(dum) 
        #             ax.fill_between(x, ci[0], ci[1] , color=pal[i], alpha=.1) 
                    
        #         # add_stim_to_plot(ax) 
        #         ax.set_xlim([0, gv.t_test[1]+1]) 
                    
        #     ax.set_ylabel('PC {}'.format(n_pc+pc_shift+1)) 
        #     ax.set_xlabel('Time (s)') 
        #     sns.despine(right=True, top=True)
        #     # if n_pc == np.amin([gv.n_components,3])-1: 
        #         # add_orientation_legend(ax) 
