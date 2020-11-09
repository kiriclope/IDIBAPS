from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac 
importlib.reload(fac) ; 
fac.SetPlotParams() 

import data.preprocessing as pp 
import data.plotting as pl 

import detrend as detrend 

scriptdir = os.path.dirname(__file__) 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
pc_shift = 0 

gv.IF_SAVE=1 
gv.DELAY_ONLY=0 
gv.DOWN_SAMPLING=0

gv.n_components = 30 #'mle' #75% => 11 C57/ChR - 18 Jaws # inflexion 2-4 
gv.correct_trial = 0  # 17-14-16 / 6 
gv.laser_on = 0 

gv.DENOISED=1
gv.AVG_BL_TRIALS = 0 
gv.perF0 = 0  
gv.DF_ONLY = 0 
gv.normFluo = 0

gv.detrend = 0 
POLY_DEG = 7 

gv.data_type= 'fluo' 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 

        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=0.0) 

        if gv.DELAY_ONLY: 
            gv.bin_start = gv.bins_delay[0] 
            gv.t_start = gv.t_ED[0] 
        
        if gv.AVG_BL_TRIALS: 
            # compute F0 over all trials 
            X = pp.dFF0(X) 
        
        # if IF_DETREND: 
        #     X_trend = [] 
        #     for n_trial in range(X.shape[0]): 
        #         fit_values = detrend.detrend_data(X[n_trial], poly_fit=1, degree=POLY_DEG) 
        #         X_trend.append(fit_values) 
        #     X_trend = np.asarray(X_trend) 
        #     X = X - X_trend[:,np.newaxis,:] 
        
        trials = [] 
        trials_avg = [] 
        for gv.trial in gv.trials: 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1) 
            
            #compute F0 only on the given trial type, eventually average over all trials in trial. 
            # if not gv.AVG_BL_TRIALS and not gv.perF0: 
            #     X_S1 = pp.dFF0(X_S1) 
            #     X_S2 = pp.dFF0(X_S2) 

            # F0_S1 = pp.estimate_F0(X_S1[:,:,gv.bins_BL]).reshape(X_S1.shape[0], X_S1.shape[1], X_S1.shape[2]) 
            # F0_S2 = pp.estimate_F0(X_S2[:,:,gv.bins_BL]).reshape(X_S2.shape[0], X_S2.shape[1], X_S2.shape[2]) 

            X_S1 = pp.deconvDF(X_S1).reshape(X_S1.shape[0], X_S1.shape[1], X_S1.shape[2]) 
            X_S2 = pp.deconvDF(X_S2).reshape(X_S2.shape[0], X_S2.shape[1], X_S2.shape[2]) 
            
            if gv.DOWN_SAMPLING:
                X_S1 = pp.downsample(X_S1,1,2) 
                X_S2 = pp.downsample(X_S2,1,2) 
                gv.trial_size = X_S1.shape[2] 
                
            if gv.perF0: 
            # # compute F0 as 5th percentile of the baseline F distribution
                F0_S1 = pp.findBaselineF0(X_S1[:,:], gv.frame_rate) 
                X_S1 = (X_S1 - F0_S1)/(F0_S1 + gv.eps) 
            
                F0_S2 = pp.findBaselineF0(X_S2[:,:], gv.frame_rate) 
                X_S2 = (X_S2 - F0_S2)/(F0_S2 + gv.eps) 
                        
            if gv.normFluo : 
                for trial in range(X_S1.shape[0]): 
                    X_S1[trial] = pp.z_score(X_S1[trial]) 
                    X_S2[trial] = pp.z_score(X_S2[trial]) 
            
            if gv.detrend: 
                X_trend = [] 
                for n_trial in range(X_S1.shape[0]): 
                    fit_values = detrend.detrend_data(X_S1[n_trial], poly_fit=1, degree=POLY_DEG) 
                    X_trend.append(fit_values) 
                X_trend = np.asarray(X_trend) 
                
                X_S1 = X_S1 - X_trend 
                # X_S1 = X_S1 - X_trend[:,np.newaxis,:] 
                
                X_trend = [] 
                for n_trial in range(X_S2.shape[0]): 
                    fit_values = detrend.detrend_data(X_S2[n_trial], poly_fit=1, degree=POLY_DEG) 
                    X_trend.append(fit_values) 
                X_trend = np.asarray(X_trend)
                
                X_S2 = X_S2 - X_trend
                # X_S2 = X_S2 - X_trend[:,np.newaxis,:] 
                
            if gv.DELAY_ONLY:
                X_S1 = X_S1[:,:, gv.bins_delay] 
                X_S2 = X_S2[:,:, gv.bins_delay]

                # gv.n_trials = 2*X_S1.shape[0]
                gv.trial_size = X_S1.shape[2]
            
            X_S1_S2 = np.vstack((X_S1, X_S2)) 
            
            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
            if 'all' in gv.samples: 
                # concatenate S1 and S2 trials 
                X_S1_S2_avg = np.mean(X_S1_S2, axis=0) 
            else: 
                # keep S1 and S2 trials separated 
                X_S1_avg = np.mean(X_S1, axis=0) 
                X_S2_avg = np.mean(X_S2, axis=0) 
                X_S1_S2_avg = np.hstack((X_S1_avg, X_S2_avg)) 
            
            print('X_S1_S2', X_S1_S2.shape) 
            print('X_S1_S2_avg', X_S1_S2_avg.shape) 

            trials.append(X_S1_S2) 
            trials_avg.append(X_S1_S2_avg) 
        
        X_trials = np.vstack(trials) 
        print('X_trials', X_trials.shape) 
        
        X_avg = np.hstack(trials_avg) 
        scaler = StandardScaler(with_mean=True, with_std=True) 
        X_avg = scaler.fit_transform(X_avg.T).T 
        print('X_avg', X_avg.shape) 
        
        pca = PCA(n_components=gv.n_components) 
        X_avg = pca.fit_transform(X_avg.T).T
        X_avg = X_avg.reshape(gv.n_components, len(gv.trials), len(gv.samples), gv.trial_size)
        X_avg = np.moveaxis(X_avg, 0, 2) 
        
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]) 

        projected_trials = [] 
        for i in range(X_trials.shape[0]): 
            # scale every trial using the same scaling applied to the averages 
            trial = scaler.transform(X_trials[i,:,:].T).T 
            # project every trial using the pca fit on averages 
            proj_trial = pca.transform(trial.T).T 
            projected_trials.append(proj_trial) 
        
        X_proj = np.asarray(projected_trials) 
        X_proj = np.reshape(X_proj, (len(gv.trials), len(gv.samples), int(gv.n_trials/len(gv.samples)), gv.n_components, gv.trial_size)) 
        
        X_trials = np.reshape(X_trials, (len(gv.trials), len(gv.samples) , int(gv.n_trials/len(gv.samples)), gv.n_neurons, gv.trial_size)) 
        
        print(X_proj.shape) 

        figname = '%s_%s_pca_scree_plot' % (gv.mouse, gv.session)
        plt.figure(figname)
        plt.plot(explained_variance,'-o')
        plt.xlabel('components')
        plt.ylabel('explained variance')
        figdir = pl.figDir(scriptdir) 
        pl.save_fig(figname) 
        
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
                    
        #         add_stim_to_plot(ax) 
        #         ax.set_xlim([0, gv.t_test[1]+1]) 
                    
        #     ax.set_ylabel('PC {}'.format(n_pc+pc_shift+1)) 
        #     ax.set_xlabel('Time (s)') 
        #     sns.despine(right=True, top=True)
        #     if n_pc == np.amin([gv.n_components,3])-1: 
        #         add_orientation_legend(ax) 

        # figdir = pl.figDir(scriptdir) 
        # pl.save_fig(figname, figdir) 
