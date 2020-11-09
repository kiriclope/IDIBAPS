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
from plotting import * 

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
# gv.samples = ['all'] 

gv.trials = ['ND','D1','D2'] 
gv.laser_on = 0
gv.n_components = 10 
pc_shift = 0
AVG_EPOCHS = 0 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_frame_rate() 
        data.get_bins(t_start=0) 
        
        # F0 = np.mean(np.mean(X[:,:,gv.bins_baseline],axis=2), axis=0)
        # F0 = F0[np.newaxis,:, np.newaxis] 

        # # idx = np.where(F0<=0.01)[1] 
        # # F0 = np.delete(F0, idx, axis=1) 
        # # X = np.delete(X, idx, axis=1) 

        # X = (X-F0) / (F0 + gv.eps) 
        
        trial_averages = []
        for gv.trial in gv.trials:
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1) 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape) 
            X_S1 = pp.dFF0(X_S1)
            X_S2 = pp.dFF0(X_S2)
            
            if AVG_EPOCHS : 
                X_S1 = data.avgOverEpochs(X_S1) 
                X_S2 = data.avgOverEpochs(X_S2) 
                gv.trial_size = len(gv.epochs) 
                
            # keep S1 and S2 trials separated                
            X_S1 = np.mean(X_S1, axis=0) 
            X_S2 = np.mean(X_S2, axis=0) 
            X_S1_S2 = np.hstack((X_S1, X_S2))         
            print('X_S1_S2', X_S1_S2.shape)
            
            trial_averages.append(X_S1_S2) 
        
        X_avg = np.hstack(trial_averages) 
        # X_avg = pp.z_score(X_avg) # Xav_sc = center(Xav)
        X_avg = pp.normalize(X_avg) # Xav_sc = center(Xav)
        print('X_avg', X_avg.shape) 
    
        pca = PCA(n_components=gv.n_components) 
        X_avg_pc = pca.fit_transform(X_avg.T).T 
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]) 

        X_avg_pc = np.asarray(X_avg_pc) 
        print(X_avg_pc.shape)

        # X_proj = np.empty( (len(gv.trials), len(gv.samples), gv.n_components, gv.trial_size) )
        # print(X_proj.shape)
        
        # for trial in range(len(gv.trials)):
        #     for sample in range(len(gv.samples)):
        #         for n_pc in range(gv.n_components):
        #             K = trial*len(gv.trial) + sample
        #             print(K)
        #             X_proj[trial,sample,n_pc] = X_avg_pc[n_pc, gv.trial_size*K:gv.trial_size*(K+1)] 

        X_avg_pc = X_avg_pc.reshape(gv.n_components, len(gv.trials), len(gv.samples), gv.trial_size)        
        print(X_avg_pc.shape)
        
        fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
        for pc in range(3):
            ax = axes[pc] 
            for i, trial in enumerate(gv.trials):
                for j, sample in enumerate(gv.samples):
                    y = X_avg_pc[pc+pc_shift, i, j] 
                    # y = X_proj[i, j, pc] 
                    y = gaussian_filter1d(y, sigma=3) 

                    ax.plot(gv.time, y, c=pal[i], label= trial+'_'+sample) 
                    ax.set_xlim([0, gv.t_test[1]+1])
                    
                    add_stim_to_plot(ax)
                    ax.set_ylabel('PC {}'.format(pc+pc_shift+1))
        add_orientation_legend(axes[2])
        axes[1].set_xlabel('Time (s)')
        sns.despine(fig=fig, right=True, top=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        add_orientation_legend(axes[2])
