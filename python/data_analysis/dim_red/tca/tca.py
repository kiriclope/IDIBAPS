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
importlib.reload(pp) ; 

import data.plotting as pl
importlib.reload(pl) ; 

import tensortools as tt

pal = ['r','b','y'] 
gv.samples = ['S1', 'S2']
gv.trials = ['ND', 'D1', 'D2']
pc_shift = 0

n_components = 100
gv.correct_trial = 0  # 17-14-16 / 6
gv.laser_on = 0

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    trials = [] 
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 

        data.get_delays_times()  
        data.get_frame_rate() 
        data.get_bins(t_start=0) 
        
        F0 = np.mean(X[:,:,gv.bins_baseline],axis=2) 
        F0 = F0[:,:, np.newaxis] 

        # F0 = np.mean(np.mean(X[:,:,gv.bins_baseline],axis=2), axis=0) 
        # F0 = F0[np.newaxis,:, np.newaxis] 
        
        # idx = np.where(F0<=0.01)[1] 
        # F0 = np.delete(F0, idx, axis=1) 
        # X = np.delete(X, idx, axis=1) 
        
        X = (X - F0) / (F0 + gv.eps)
        
        for gv.trial in gv.trials: 
            X_S1, X_S2 = data.get_S1_S2_trials(X, y) 
            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)             
            data.get_trial_types(X_S1) 

            X_S1_S2 = np.vstack((X_S1, X_S2)) 
            X_S1_S2 = X_S1_S2[:,:]
            print('X_S1_S2', X_S1_S2.shape)
            
            trials.append(X_S1_S2) 
        
    X_trials = np.vstack(trials)
    print('X_trials', X_trials.shape) 

    for trial in range(X_trials.shape[0]):
        X_trials[trial] = pp.normalize(X_trials[trial]) 
        
    # bins = np.arange(gv.bins_ED[0], gv.bins_LD[-1])
    
    # X_trials = X_trials[:,:, bins] 
    # # X_trials = gaussian_filter1d(X_trials,sigma=3) 
    print('X_trials', X_trials.shape)
        
    data = np.moveaxis(X_trials, 0, 2) 
    print('tensor', data.shape) 
    
    # Fit an ensemble of models, 4 random replicates / optimization runs per model rank 
    ensemble = tt.Ensemble(fit_method="ncp_hals") 
    ensemble.fit(data, ranks=range(1, n_components), replicates=2) 

    fig, axes = plt.subplots(1, 2)  
    tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
    tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    fig.tight_layout()

    # # # Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
    num_components = np.amin([n_components-1, 10])
    # replicate = 0 
    # tt.plot_factors(ensemble.factors(num_components)[replicate], plots=['bar','line','scatter'])  # plot the low-d factors 

    # fig = plt.gcf()
    # for axis in [1,4,7,10,13,16]:
    #     ax = fig.axes[axis] 
    #     pl.add_stim_to_plot(ax, bin_start=gv.bins_ED[0])
    #     # ax.set_xlim([0, gv.bins_test[1]+1])

    neuron_factors = ensemble.factors(num_components)[0][0]
    time_factors = ensemble.factors(num_components)[0][1]
    trial_factors = ensemble.factors(num_components)[0][2]

    # X_tca = np.empty( [neuron_factors.shape[0], time_factors.shape[0], trial_factors.shape[0]] )
    # for n in range(neuron_factors.shape[0]):
    #     for t in range(time_factors.shape[0]):
    #         for k in range(trial_factors.shape[0]):
    #             for r in range(trial_factors.shape[1]):
    #                 X_tca[n,t,k] = np.sum(neuron_factors[n]*time_factors[t]*trial_factors[k])

    X_tca = np.empty( [trial_factors.shape[0], trial_factors.shape[1], time_factors.shape[0]] )
    for k in range(trial_factors.shape[0]):
        for t in range(time_factors.shape[0]):
            for r in range(trial_factors.shape[1]):
                X_tca[k,r,t] = trial_factors[k,r]*time_factors[t,r]

