from libs import * 

sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac 
importlib.reload(fac) ; 
fac.SetPlotParams() 

import data.preprocessing as pp
import data.plotting as pl 

from matplotlib.ticker import PercentFormatter 

pal = ['r','b','y'] 
gv.data_type = 'fluo' 

for gv.mouse in [gv.mice[1]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times()
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        data.get_delays_times() 
        data.get_bins(t_start=0) 

        bool_ND = (y_labels[4]==0) & (y_labels[8]==0)
        bool_D1 = (y_labels[4]==13) & (y_labels[8]==0)
        bool_D2 = (y_labels[4]==14) & (y_labels[8]==0)

        bool_correct = ( y_labels[2]==1 ) & ( y_labels[2]==4 ) 
        
        ND_trials = np.argwhere( bool_ND ).flatten() 
        D1_trials = np.argwhere( bool_D1 ).flatten() 
        D2_trials = np.argwhere( bool_D2 ).flatten() 

        ND_correct = np.argwhere( bool_ND & bool_correct ).flatten() 
        D1_correct = np.argwhere( bool_D1 & bool_correct ).flatten() 
        D2_correct = np.argwhere( bool_D2 & bool_correct ).flatten() 

        print(ND_correct)

