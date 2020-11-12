from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) 

import data.utils as data 
importlib.reload(data) 

import data.fct_facilities as fac 
importlib.reload(fac)  
fac.SetPlotParams() 

import data.preprocessing as pp 
import data.plotting as pl 

import detrend 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
pc_shift = 0 

gv.IF_SAVE=0 

gv.DENOISED=0 
gv.DELAY_ONLY=0
gv.ED_MD_LD = 0
gv.DOWN_SAMPLING=0 

gv.detrend = 0 
POLY_DEG = 7 

gv.correct_trial = 0  # 17-14-16 / 6 
gv.laser_on = 0 

gv.n_components = 50 #'mle' #75% => 11 C57/ChR - 18 Jaws # inflexion 2-4 
gv.data_type= 'fluo' 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 

        data.get_delays_times() 
        data.get_bins(t_start=0.0) 
            
        if gv.mouse in [gv.mice[0]]: 
            gv.n_trials = 40 
        else: 
            gv.n_trials = 32 

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
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1) 
            
            X_S1 = pp.dFF0(X_S1) 
            X_S2 = pp.dFF0(X_S2) 
            
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
                
            # for i in range(X_S1.shape[0]): 
            #     #     #     X_S1[i] = pp.z_score(X_S1[i]) 
            #     #     #     X_S2[i] = pp.z_score(X_S2[i]) 
            #     X_S1[i] = pp.normalize(X_S1[i]) 
            #     X_S2[i] = pp.normalize(X_S2[i])
            
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
            
            X_trials[n_trial,0] = X_S1
            X_trials[n_trial,1] = X_S2
            
            X_avg[n_trial] = np.hstack( ( np.mean(X_S1, axis=0), np.mean(X_S2, axis=0) ) ) 
            
        print('X_trials', X_trials.shape)         

        X_avg = np.hstack(X_avg)
        print('X_avg', X_avg.shape) 
        
        scaler = StandardScaler(with_mean=True, with_std=True) 
        X_avg = scaler.fit_transform(X_avg.T).T 
        
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

        figname = '%s_%s_pca_scree_plot' % (gv.mouse, gv.session) 
        plt.figure(figname) 
        plt.plot(explained_variance,'-o') 
        plt.xlabel('components')
        plt.ylabel('explained variance') 
        # figdir = pl.figDir(scriptdir) 
        # pl.save_fig(figname)         
