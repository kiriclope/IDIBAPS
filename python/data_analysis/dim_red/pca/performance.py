from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
import data.utils as data 

import data.fct_facilities as fac 
fac.SetPlotParams() 

import data.preprocessing as pp
import data.plotting as pl 

from matplotlib.ticker import PercentFormatter 

pal = ['r','b','y'] 
gv.data_type = 'fluo' 

for gv.mouse in gv.mice : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()

    perf = np.empty( ( len(gv.trials), len(gv.sessions) ) ) 
    
    for n_session, gv.session in enumerate(gv.sessions) : 
        X, y_labels = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y_labels.shape) 
        
        data.get_delays_times() 
        data.get_bins(t_start=0) 

        bool_ND = (y_labels[4]==0) & (y_labels[8]==0)
        bool_D1 = (y_labels[4]==13) & (y_labels[8]==0)
        bool_D2 = (y_labels[4]==14) & (y_labels[8]==0)

        # print(bool_ND) 
        # print(bool_D1) 
        # print(bool_D2) 
        
        bool_correct = ( y_labels[2]==1 ) | ( y_labels[2]==4 ) 
        
        ND_trials = len( np.argwhere( bool_ND ).flatten() )
        D1_trials = len( np.argwhere( bool_D1 ).flatten() )
        D2_trials = len( np.argwhere( bool_D2 ).flatten() )

        # print(ND_trials) 
        # print(D1_trials) 
        # print(D2_trials) 
        
        ND_correct = len( np.argwhere( bool_ND & bool_correct ).flatten() )
        D1_correct = len( np.argwhere( bool_D1 & bool_correct ).flatten() )
        D2_correct = len( np.argwhere( bool_D2 & bool_correct ).flatten() )

        # print(ND_correct/ND_trials*100) 
        # print(D1_correct/D1_trials*100) 
        # print(D2_correct/D2_trials*100) 

        perf[0, n_session] = ND_correct/ND_trials*100 
        perf[1, n_session] = D1_correct/D1_trials*100 
        perf[2, n_session] = D2_correct/D2_trials*100 

    figtitle = '%s_performance' % (gv.mouse)
    ax = plt.figure(figtitle).add_subplot() 
    
    days = np.arange( len(gv.sessions) ) 
    plt.plot(days, perf[0], color=pal[0]) 
    plt.plot(days, perf[1], color=pal[1]) 
    plt.plot(days, perf[2], color=pal[2])
    ax.set_title(gv.mouse)
    plt.xlabel('days')
    plt.ylabel('performance')
    plt.ylim([50,100])
    plt.xlim([0,5])
    
    pl.figDir() 
    pl.save_fig(figtitle) 
