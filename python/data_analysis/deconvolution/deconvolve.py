from libs import * 
sys.path.insert(1, '/homecentral/alexandre.mahrach/gdrive/postdoc_IDIBAPS/python/data_analysis') 

import data.constants as gv 
importlib.reload(gv) ; 

import data.utils as data 
importlib.reload(data) ; 

import data.fct_facilities as fac 
importlib.reload(fac) ; 
fac.SetPlotParams() 

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

gv.correct_trial = 0 
gv.laser_on = 0 
gv.data_type= 'fluo' 

for gv.mouse in [gv.mice[2]] : 

    data.get_sessions_mouse() 
    data.get_stimuli_times() 
    data.get_delays_times() 
    
    for gv.session in [gv.sessions[-1]] : 
        X, y = data.get_fluo_data() 
        print('mouse', gv.mouse, 'session', gv.session, 'data X', X.shape,'y', y.shape) 
        
        a = constrained_foopsi(X[0,0],p=2) 
        plt.plot(X[0,0])
        plt.plot(a[0])
