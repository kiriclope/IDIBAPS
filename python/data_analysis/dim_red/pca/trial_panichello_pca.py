from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac 
importlib.reload(fac) ; 
fac.SetPlotParams() 

import preprocessing as prep 
from plotting import * 

import detrend as detrend 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
pc_shift = 4
gv.trials= ['D1'] 

gv.n_components = 3 #'mle' #75% => 11 C57/ChR - 18 Jaws 
gv.correct_trial = 0  # 17-14-16 / 6
gv.laser_on = 0

IF_DETREND = 0 
POLY_DEG = 1

for gv.mouse in [gv.mice[0]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 

        data.get_delays_times()  
        data.get_frame_rate() 
        data.get_bins(t_start=0.5) 
        
        # F0 = np.mean(X[:,:,gv.bins_baseline],axis=2) 
        # F0 = F0[:,:, np.newaxis] 
        
        F0 = np.mean(np.mean(X[:,:,gv.bins_baseline],axis=2), axis=0)
        F0 = F0[np.newaxis,:, np.newaxis]
        
        X = (X - F0) / (F0 + gv.eps) 

        if IF_DETREND: 
            X_trend = [] 
            for n_trial in range(X.shape[0]): 
                fit_values = detrend.detrend_data(X[n_trial], poly_fit=1, degree=POLY_DEG) 
                X_trend.append(fit_values) 
            X_trend = np.asarray(X_trend) 

            X = X - X_trend[:,np.newaxis,:] 
            
        trials = [] 
        trials_avg = []
        X_S1_S2_avg = [] 
        for gv.trial in gv.trials: 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1)

            # data with each trials
            X_S1_S2 = np.vstack((X_S1, X_S2))
            trials.append(X_S1_S2)
            
            # average over trials 
            # keep S1 and S2 trials separated 
            X_S1_avg = np.mean(X_S1, axis=0) 
            X_S2_avg = np.mean(X_S2, axis=0) 
            trials_avg.append([X_S1_avg, X_S2_avg])
             

        X_trials = np.vstack(trials) 
        X_avg = np.vstack(trials_avg) 
        print('X_trials', X_trials.shape, 'X_avg', X_avg.shape) 
        
        # average over conditions (samples/trials) 
        mean_cols = np.mean(X_avg, axis=0) 
        X_avg = X_avg - mean_cols[np.newaxis,:,:] 
        print('X_avg', X_avg.shape) 

        X_pc = np.hstack(X_avg) 
        print('X_pc', X_pc.shape) 
        
        scaler = StandardScaler(with_mean=True, with_std=True) 
        X_pc = scaler.fit_transform(X_pc.T).T         
        pca = PCA(n_components=gv.n_components) 
        pca.fit_transform(X_pc.T) 
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_
        print('n_pc', gv.n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]) 
    
        projected_trials = []
        for i in range(X_avg.shape[0]): 
            # scale every trial using the same scaling applied to the averages 
            trial = scaler.transform(X_avg[i,:,:].T).T 
            # trial = scaler.transform(X_trials[i,:,:].T).T 
            # project every trial using the pca fit on averages 
            proj_trial = pca.transform(trial.T).T 
            projected_trials.append(proj_trial) 
        
        X_proj = np.asarray(projected_trials)
        X_proj = X_proj.reshape(len(gv.trials), len(gv.samples), gv.n_components, gv.trial_size) 
        # X_proj = np.reshape(X_proj, (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_components, gv.trial_size)) 
        print(X_proj.shape) 
    
        if gv.laser_on:
            figname = '%s_%s_pca_laser_on_%d' % (gv.mouse, gv.session, pc_shift)
        else:
            figname = '%s_%s_pca_laser_off_%d' % (gv.mouse, gv.session, pc_shift)

        plt.figure(figname, figsize=[10, 2.8])    
        x = gv.time 
        for n_pc in range(np.amin([gv.n_components,3])): 
            ax = plt.figure(figname).add_subplot(1, 3, n_pc+1) 
            for i, trial in enumerate(gv.trials): 
                for j, sample in enumerate(gv.samples): 
                    # dum = X_proj[i,j,:,n_pc+pc_shift,:].transpose() 
                    # y = np.mean( dum, axis=1) 
                    y = X_proj[i,j,n_pc,:].transpose() 
                    y = gaussian_filter1d(y, sigma=1)  
                    ax.plot(x, y, color=pal[i]) 
                    
                    ax.plot(x, y, color=pal[i]) 
                    # ci = prep.conf_inter(dum) 
                    # ax.fill_between(x, ci[0], ci[1] , color=pal[i], alpha=.1) 
                add_stim_to_plot(ax) 
                ax.set_xlim([0, gv.t_test[1]+1]) 
                
            ax.set_ylabel('PC {}'.format(n_pc+1)) 
            ax.set_xlabel('Time (s)') 
            sns.despine(right=True, top=True)
            if n_pc == np.amin([gv.n_components,3])-1: 
                add_orientation_legend(ax) 

        figdir = figDir() 
        save_fig(figname, figdir) 
