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
n_components=3
gv.trials = ['ND','D1','D2'] 

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

        # F0 = np.mean(X[:,:,gv.bins_baseline],axis=2)
        # F0 = F0[:,:, np.newaxis]
        
        F0 = np.mean(np.mean(X[:,:,gv.bins_baseline],axis=2), axis=0)
        F0 = F0[np.newaxis,:, np.newaxis]         
        X = (X -F0) / (F0 + gv.eps) 
        
        trials = []
        for gv.trial in gv.trials:
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            data.get_trial_types(X_S1)
            
            trial_type = ['ND'] * gv.n_trials + ['D1'] * gv.n_trials + ['D2'] * gv.n_trials 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)

            X_S1_S2 = np.hstack( (np.hstack(X_S1), np.hstack(X_S2)) ) 
            print('X_S1_S2', X_S1_S2.shape)

            trials.append(X_S1_S2)
            
        Xl = np.hstack(trials) 
        # Xl = pp.z_score(Xl) 
        Xl = pp.normalize(Xl) 
        print('Xl', Xl.shape)
        
        pca = PCA(n_components=n_components)
        Xl_p = pca.fit_transform(Xl.T).T
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance, 'total' , np.cumsum(explained_variance)[-1]*100) 
        
        gt = {comp : {t_type : [] for t_type in gv.trials} for comp in range(n_components)}

        for comp in range(n_components):
            for i, t_type in enumerate(trial_type):
                t = Xl_p[comp, gv.trial_size * i: gv.trial_size * (i + 1)]
                gt[comp][t_type].append(t)

        X_proj = []
        for comp in range(n_components):
            for t_type in gv.trials: 
                gt[comp][t_type] = np.vstack(gt[comp][t_type]).transpose() 
                gt[comp][t_type] = gt[comp][t_type].reshape(gv.trial_size, len(gv.samples), int(gv.n_trials/len(gv.samples))) 
                X_proj.append(gt[comp][t_type])

        X_proj=np.asarray(X_proj).reshape( len(gv.trials), n_components, gv.trial_size, len(gv.samples), int(gv.n_trials/len(gv.samples)))
        X_proj = np.moveaxis(X_proj, 0,4) 
        X_proj = np.moveaxis(X_proj, 1,4) 
        print(X_proj.shape)
        
        f, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey=True, sharex=True) 
        x = gv.time
        for comp in range(3):
            ax = axes[comp]
            for t, t_type in enumerate(gv.trials):
                for i  in range(2):
                    y = np.mean(gt[comp][t_type][:,i,:],axis=1) 
                    ax.plot(x, y, color=pal[t]) 
                    ci = stats.t.interval(0.95, len(y)-1, loc=np.mean(y), scale=stats.sem(y)) 
                    ax.fill_between(x, y-ci[0], y+ci[1] , color=pal[t], alpha=.1) 
                add_stim_to_plot(ax) 
                ax.set_xlim([0, gv.t_test[1]+1])
            ax.set_ylabel('PC {}'.format(comp+1))
            axes[1].set_xlabel('Time (s)')
            sns.despine(right=True, top=True)
            add_orientation_legend(axes[2])
