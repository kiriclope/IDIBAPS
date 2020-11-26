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
import data.synthetic as syn
import data.plotting as pl

pal = ['r','b','y']
pal = ['r','b','y'] 
gv.samples = ['S1', 'S2'] 
pc_shift = 0 
gv.pca_concat = 1

gv.IF_PCA=1
gv.explained_variance = 0.20

gv.DELAY_ONLY=0
gv.ED_MD_LD=0
SYN = 0

gv.n_components= 160 # 90% 100 
gv.standardize=1 
gv.IF_SAVE = 1

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
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data()
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_bins(t_start=0) 

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

        
        trials = [] 
        for n_trial, gv.trial in enumerate(gv.trials):
            if SYN:
                X_S1 = X_syn[n_trial,0]
                X_S2 = X_syn[n_trial,1] 
            else: 
                X_S1, X_S2 = data.get_S1_S2_trials(X, y)            
            data.get_trial_types(X_S1)
            
            X_S1 =pp.dFF0(X_S1)
            X_S2 =pp.dFF0(X_S2)
            
            if gv.DELAY_ONLY: 
                X_S1 = X_S1[:,:, gv.bins_delay] 
                X_S2 = X_S2[:,:, gv.bins_delay] 
            elif gv.ED_MD_LD: 
                X_S1 = X_S1[:,:, gv.bins_ED_MD_LD] 
                X_S2 = X_S2[:,:, gv.bins_ED_MD_LD] 

            print('X_S1', X_S1.shape, 'X_S2', X_S2.shape)

            X_S1_S2 = np.hstack( (np.hstack(X_S1), np.hstack(X_S2)) ) 
            print('X_S1_S2', X_S1_S2.shape) 

            trials.append(X_S1_S2)
            
        X_concat = np.hstack(trials) 
        # X_concat = pp.z_score(X_concat)
        # X_concat = pp.normalize(X_concat) 

        # standardize neurons/features across trials/samples
        scaler = StandardScaler(with_mean=True, with_std=True) 
        scaler.fit(X_concat.T) 
        X_concat = scaler.transform(X_concat.T).T 
        
        gv.n_components = get_optimal_number_of_components(X_concat) 
        pca = PCA(n_components=gv.n_components)
        # pca on X: trials x neurons
        X_concat = pca.fit_transform(X_concat.T).T
        
        print('X_concat', X_concat.shape) 
        
        explained_variance = pca.explained_variance_ratio_ 
        gv.n_components = pca.n_components_ 
        print('n_pc', gv.n_components,'explained_variance', explained_variance[0:3], 'total' , np.cumsum(explained_variance)[-1]*100) 
        
        X_proj = np.empty( ( len(gv.trials), len(gv.samples), int( gv.n_trials/len(gv.samples) ), gv.n_components , gv.trial_size) ) 
        for i in range( len(gv.trials) ): 
            for j in range( len(gv.samples) ) : 
                for k in range( int( gv.n_trials/len(gv.samples) ) ) : 
                    for l in range(gv.n_components) : 
                        m = i*len(gv.samples)* int( gv.n_trials/len(gv.samples) ) + j * int( gv.n_trials/len(gv.samples) )  + k 
                        X_proj[i,j,k,l] = X_concat[l, gv.trial_size * m: gv.trial_size * (m + 1)].flatten() 
            
        print(X_proj.shape)        

        figname = '%s_%s_pca_scree_plot' % (gv.mouse, gv.session) 
        plt.figure(figname) 
        plt.plot(explained_variance,'-o') 
        plt.xlabel('components')
        plt.ylabel('explained variance') 
        figdir = pl.figDir() 
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
                    
        #         # add_stim_to_plot(ax) 
        #         ax.set_xlim([0, gv.t_test[1]+1]) 
                    
        #     ax.set_ylabel('PC {}'.format(n_pc+pc_shift+1)) 
        #     ax.set_xlabel('Time (s)') 
        #     sns.despine(right=True, top=True)
        #     if n_pc == np.amin([gv.n_components,3])-1: 
        #         add_orientation_legend(ax) 
