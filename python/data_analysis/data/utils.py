from .libs import *
from . import constants as gv

def get_delays_times():
    if((gv.mouse=='ChRM04') | (gv.mouse=='JawsM15')):
        gv.t_ED = [3, 4.5]
        gv.t_MD = [5.5, 6.5]
        gv.t_LD = [7.5, 9]
    else:
        gv.t_ED = [3, 6]
        gv.t_MD = [7, 8]
        gv.t_LD = [9, 12]

def get_stimuli_times():
    if((gv.mouse=='ChRM04') | (gv.mouse=='JawsM15')):
        gv.t_BL = [0, 2]
        gv.t_STIM = [2, 3]
        gv.t_DIST = [4.5, 5.5]
        gv.t_cue = [6.5, 7]
        gv.t_DRT_reward = [7, 7.5]
        gv.t_test = [9, 10]
    else:
        gv.t_BL = [0, 2]
        gv.t_STIM = [2, 3]
        gv.t_DIST = [6, 7]
        gv.t_cue = [8, 8.5]
        gv.t_DRT_reward = [8.5, 9]
        gv.t_test = [12, 13]

def get_frame_rate():
    if((gv.mouse=='ChRM04') | (gv.mouse=='JawsM15')):
        gv.frame_rate = 6
    else :
        gv.frame_rate = 7.5

def get_sessions_mouse():
    if gv.mouse=='C57_2_DualTask' :
        gv.sessions = list( map( str, np.arange(20200116, 20200121) ) )
    elif gv.mouse=='ChRM04' :
        gv.sessions = list( map( str, np.arange(20200521, 20200527) ) )
    elif gv.mouse=='JawsM15' :
        gv.sessions = list( map( str, np.arange(20200605, 20200610) ) )

def get_fluo_data():

    if((gv.mouse=='ChRM04') | (gv.mouse=='JawsM15')):
        data = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.mouse + '/' + gv.session + 'SumFluoTraceFile' + '.mat')

        if 'rates' in gv.data_type:
            X_data = np.rollaxis(data['S_dec'],1,0)
        else:
            X_data = np.rollaxis(data['C_df'], 1,0) 
            # X_data = np.rollaxis(data['dFF0'],1,0)

        y_labels = data['Events'].transpose()
        gv.frame_rate = 6
    
    else:
        data = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.mouse +  '/' + gv.session + '-C57-2-DualTaskAcrossDaySameROITrace' + '.mat')
        data_labels = loadmat('/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/DataForAlexandre/' + gv.mouse + '/' + gv.session + '-C57-2-DualTask-SumFluoTraceFile' + '.mat')
        
        X_data = np.rollaxis(data['SameAllCdf'],2,0)
        # X_data = np.rollaxis(data['SamedFF0'],2,0)
        
        y_labels= data_labels['AllFileEvents'+gv.session][0][0][0].transpose() 
        gv.frame_rate = 7.5 

    gv.duration = X_data.shape[2]/gv.frame_rate
    gv.time = np.linspace(0,gv.duration,X_data.shape[2]);
    gv.bins = np.arange(0,len(gv.time))
    gv.n_neurons = X_data.shape[1]
    
    return X_data, y_labels

def which_trials(y_labels):
    y_trials = []

    bool_correct = (y_labels[2]==1)
    if 'ND' in gv.trial:
        if 'S1' in gv.trial:
            if gv.laser_on:                
                y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]!=0) ).flatten()            
            if not gv.laser_on:
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten() 
                    
        elif 'S2' in gv.trial: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==0) & (y_labels[8]==0) ).flatten()                    
        else: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) & bool_correct ).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[4]==0) & (y_labels[8]==0) ).flatten()
                    
    elif 'D1' in gv.trial: 
        if 'S1' in gv.trial: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten() 
        elif 'S2' in gv.trial: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]==0) & bool_correct ).flatten() 
                else:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==13) & (y_labels[8]==0) ).flatten() 
        else:
            if gv.laser_on: 
                y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]!=0) ).flatten()
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0) & bool_correct).flatten() 
                else:
                    y_trials = np.argwhere((y_labels[4]==13) & (y_labels[8]==0) ).flatten() 
    elif 'D2' in gv.trial: 
        if 'S1' in gv.trial: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
        elif 'S2' in gv.trial:
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==14) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==14) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[4]==14) & (y_labels[8]==0) ).flatten()
        else: 
            if gv.laser_on: 
                y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0) & bool_correct).flatten() 
                else:
                    y_trials = np.argwhere((y_labels[4]==14) & (y_labels[8]==0) ).flatten()
    elif 'all' in gv.trial:
        if 'S1' in gv.trial:
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[8]==0) & bool_correct).flatten()
                else:
                    y_trials = np.argwhere( (y_labels[0]==17) & (y_labels[8]==0) ).flatten()
        elif 'S2' in gv.trial: 
            if gv.laser_on: 
                y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[8]!=0) ).flatten() 
            if not gv.laser_on: 
                if gv.correct_trial:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[8]==0) & bool_correct).flatten() 
                else:
                    y_trials = np.argwhere( (y_labels[0]==18) & (y_labels[8]==0) ).flatten() 
    return y_trials

def get_S1_S2_trials(X_data, y_labels):

    trial = gv.trial
    gv.trial = trial + "_S1" 
    y_S1_trials = which_trials(y_labels) 
    # print(y_S1_trials) 
    
    gv.trial = trial + "_S2" 
    y_S2_trials = which_trials(y_labels) 
    # print(y_S2_trials) 

    gv.trial = trial 
    X_S1_trials = X_data[y_S1_trials] 
    X_S2_trials = X_data[y_S2_trials] 
    
    return X_S1_trials, X_S2_trials 

def get_trial_types(X_S1_trials): 
    gv.n_trials = 2*X_S1_trials.shape[0]
    gv.trial_type = ['ND'] * gv.n_trials + ['D1'] * gv.n_trials + ['D2'] * gv.n_trials
    # gv.trial_size = X_S1_trials.shape[2]
    # gv.t_type_ind = [np.argwhere(np.array(gv.trial_type) == t_type)[:, 0] for t_type in gv.trials]
    
def get_S1_S2_all(X_data, y_labels):

    trial = gv.trial
    gv.trial = "all_S1" 
    y_S1_all = which_trials(y_labels) 
    # print(y_S1_all) 
    
    gv.trial = "all_S2" 
    y_S2_all = which_trials(y_labels) 
    # print(y_S2_all) 

    gv.trial = trial 
    X_S1_all = X_data[y_S1_all] 
    X_S2_all = X_data[y_S2_all] 

    return X_S1_all, X_S2_all 

def get_bins(t_start=0):

    if(t_start==0): 
        gv.bins_BL = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_BL[0]) and (gv.time[bin]<=gv.t_BL[1])] 
    
        gv.bins_STIM = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_STIM[0]) and (gv.time[bin]<=gv.t_STIM[1]) ] 
    
        gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_ED[1]) ]
        
        gv.bins_DIST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DIST[0]) and (gv.time[bin]<=gv.t_DIST[1]) ]
        
        gv.bins_MD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_MD[0]) and (gv.time[bin]<=gv.t_MD[1]) ]
        
        gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_LD[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
        gv.bins_cue = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_cue[0]) and (gv.time[bin]<=gv.t_cue[1]) ] 

        gv.bins_DRT_rwd = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DRT_reward[0]) and (gv.time[bin]<=gv.t_DRT_reward[1]) ] 

        gv.bins_test = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_test[0]) and (gv.time[bin]<=gv.t_test[1]) ]

        gv.bins_delay = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 

        gv.t_delay = [ gv.time[bin] for bin in gv.bins if (gv.time[bin]>=gv.t_ED[0]) and (gv.time[bin]<=gv.t_LD[1]) ] 

    else:
        gv.bins_BL = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_BL[1]-t_start) and (gv.time[bin]<=gv.t_BL[1]) ] 
    
        gv.bins_STIM = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_STIM[1]-t_start) and (gv.time[bin]<=gv.t_STIM[1]) ] 
    
        gv.bins_ED = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_ED[1]-t_start) and (gv.time[bin]<=gv.t_ED[1]) ]
        
        gv.bins_DIST = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DIST[1]-t_start) and (gv.time[bin]<=gv.t_DIST[1]) ]
        
        gv.bins_MD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_MD[1]-t_start) and (gv.time[bin]<=gv.t_MD[1]) ]
        
        gv.bins_LD = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_LD[1]-t_start) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
        gv.bins_cue = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_cue[1]-t_start) and (gv.time[bin]<=gv.t_cue[1]) ] 

        gv.bins_DRT_rwd = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_DRT_reward[1]-t_start) and (gv.time[bin]<=gv.t_DRT_reward[1]) ] 

        gv.bins_test = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_test[1]-t_start) and (gv.time[bin]<=gv.t_test[1]) ] 

        gv.bins_delay = [ bin for bin in gv.bins if (gv.time[bin]>=gv.t_LD[1]-t_start) and (gv.time[bin]<=gv.t_LD[1]) ] 

        gv.t_delay = [ gv.time[bin] for bin in gv.bins if (gv.time[bin]>=gv.t_LD[1]-t_start) and (gv.time[bin]<=gv.t_LD[1]) ] 
        
def get_X_y_epochs(X_S1_trials, X_S2_trials): 

    X_S1 = [] 
    X_S2 = [] 

    if 'all' in gv.epochs :
        X_S1=X_S1_trials
        X_S2=X_S2_trials

    if 'Baseline' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_BL-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_BL-gv.bin_start],axis=2)) 

    if 'Stim' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_STIM-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_STIM-gv.bin_start],axis=2)) 

    if 'ED' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_ED-gv.bin_start],axis=2))
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_ED-gv.bin_start],axis=2))

    if 'Dist' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_DIST-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_DIST-gv.bin_start],axis=2)) 

    if 'MD' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_MD-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_MD-gv.bin_start],axis=2)) 
        
    if 'LD' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_LD-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_LD-gv.bin_start],axis=2)) 

    if 'Cue' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_cue-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_cue-gv.bin_start],axis=2))
        
    if 'DRT_rwd' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_DRT_rwd-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_DRT_rwd-gv.bin_start],axis=2))

    if 'Test' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_test-gv.bin_start],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_test-gv.bin_start],axis=2))
        
    X_S1 = np.asarray(X_S1)  
    X_S2 = np.asarray(X_S2) 
    
    if 'all' in gv.epochs :
        X = np.concatenate([X_S1, X_S2], axis=0) 
    else: 
        X = np.concatenate([X_S1, X_S2], axis=1) 
        X = np.rollaxis(X,2,1).transpose() 
        
    y_S1 = np.repeat(0, int(X_S1_trials.shape[0])) 
    y_S2 = np.repeat(1, int(X_S2_trials.shape[0]))
    
    y = np.concatenate((y_S1, y_S2)) 
    
    return X, y 


def avgOverEpochs(X): 

    X_avg = []
    
    if 'Baseline' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_BL])) 

    if 'Stim' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_STIM])) 

    if 'ED' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_ED]))

    if 'Dist' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_DIST])) 

    if 'MD' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_MD])) 
        
    if 'LD' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_LD])) 

    if 'Cue' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_cue])) 
        
    if 'DRT_rwd' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_DRT_rwd])) 

    if 'Test' in gv.epochs: 
        X_avg.append(np.mean(X[gv.bins_test])) 
        
    X_avg = np.asarray(X_avg)
    # X_avg = np.rollaxis(X_avg,2,1).transpose() 
    
    return X_avg

def get_X_epochs(X_trials): 

    X = [] 

    if 'all' in gv.epochs :
        X=X_trials

    if 'Baseline' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_BL],axis=2)) 

    if 'Stim' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_STIM],axis=2)) 

    if 'ED' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_ED],axis=2))

    if 'Dist' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_DIST],axis=2)) 

    if 'MD' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_MD],axis=2)) 
        
    if 'LD' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_LD],axis=2)) 

    if 'Cue' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_cue],axis=2)) 
        
    if 'DRT_rwd' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_DRT_rwd],axis=2)) 

    if 'Test' in gv.epochs: 
        X.append(np.mean(X_trials[:,:,gv.bins_test],axis=2)) 
        
    X = np.asarray(X)  
    
    X = np.rollaxis(X,2,1).transpose() 
    
    return X

def get_dX_epochs(X_S1_trials, X_S2_trials): 

    X_S1 = [] 
    X_S2 = [] 

    if 'all' in gv.epochs :
        X_S1=X_S1_trials
        X_S2=X_S2_trials

    if 'Baseline' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_BL],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_BL],axis=2)) 

    if 'Stim' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_STIM],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_STIM],axis=2)) 

    if 'ED' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_ED],axis=2))
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_ED],axis=2))

    if 'Dist' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_DIST],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_DIST],axis=2)) 

    if 'MD' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_MD],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_MD],axis=2)) 
        
    if 'LD' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_LD],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_LD],axis=2)) 

    if 'Cue' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_cue],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_cue],axis=2))
        
    if 'DRT_rwd' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_DRT_rwd],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_DRT_rwd],axis=2))

    if 'Test' in gv.epochs: 
        X_S1.append(np.mean(X_S1_trials[:,:,gv.bins_test],axis=2)) 
        X_S2.append(np.mean(X_S2_trials[:,:,gv.bins_test],axis=2))
        
    X_S1 = np.asarray(X_S1)  
    X_S2 = np.asarray(X_S2) 
    
    if 'all' in gv.epochs :
        X = np.concatenate([X_S1, X_S2], axis=0) 
    else: 
        dX = X_S1-X_S2  
        dX = np.rollaxis(dX,2,1).transpose() 
        X_S1 = np.rollaxis(X_S1,2,1).transpose() 
        X_S2 = np.rollaxis(X_S2,2,1).transpose() 
       
    return X_S1, X_S2


def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array

def get_X_y_trials(X_data, y_labels):
    X_S1_trials, X_S2_trials = get_S1_S2_trials(X_data, y_labels) 
    # print(y_S1_trials[0]) 
    
    X_S1 = bin_data(X_S1_trials, gv.n_bin, gv.n_bin) 
    X_S2 = bin_data(X_S2_trials, gv.n_bin, gv.n_bin) 
    
    # print(X_S1[0]) 
    X_trials = np.concatenate([X_S1,X_S2],axis=0) 
    # print(X_trials[0])
    y_S1 = np.repeat(0, int(X_S1_trials.shape[0]))
    y_S2 = np.repeat(1, int(X_S2_trials.shape[0]))

    y_trials = np.concatenate((y_S1, y_S2))
    
    return X_trials, y_trials

def plot_avg(X):
    time = np.linspace(0, gv.duration, X.shape[2]);
    X_window = np.mean(X, axis=1) # avg over neurons 
    X_avg = np.mean(X_window,axis=0) # avg over trials 
    X_std = np.std(X_window,axis=0) # std over trials

    plt.plot(time,X_avg,c='k')
    plt.fill_between(time, X_avg - X_std, X_avg + X_std, alpha=0.25, color='magenta')
    
    plt.xlabel('t (s)') ;
    plt.ylabel('$\Delta F$')

def vplot_delays():
    plt.gcf() 
    plt.axvline(x=gv.t_STIM[0], c='k', ls='--') # DPA sample onset 
    plt.axvline(x=gv.t_STIM[1], c='k', ls='--')

    plt.axvline(x=gv.t_test[0], c='k', ls='--') # DPA test onset
    plt.axvline(x=gv.t_test[1], c='k', ls='--')
    
    plt.axvline(x=gv.t_DIST[0], c='r', ls='--') # DRT distractor onset
    plt.axvline(x=gv.t_DIST[1], c='r', ls='--')

    plt.axvline(x=gv.t_cue[0], c='r', ls='--') # DRT cue onset
    plt.axvline(x=gv.t_cue[1], c='r', ls='--')
