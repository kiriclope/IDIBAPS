import sys 
sys.path.insert(1, '/homecentral/alexandre.mahrach/IDIBAPS/python/data_analysis') 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter 

import data.constants as gv 
import data.plotting as pl 
import data.utils as data 
import data.fct_facilities as fac 
fac.SetPlotParams() 

pal = ['r','b','y'] 
gv.data_type = 'fluo' 

figtitle = 'performance' 
fig = plt.figure(figtitle, figsize=(2.1*1.25*3, 1.85*1.25)) 

for i_mouse, gv.mouse in enumerate(gv.mice) :
    
    data.get_days()
    perf = np.empty( ( len(gv.trials), len(gv.days) ) ) 
    
    for i_day, gv.day in enumerate(gv.days): 
        X, y_labels = data.get_fluo_data() 
        
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
        
        print('correct trials', ND_correct, D1_correct, D2_correct) 
        print('error trials', ND_trials-ND_correct, D1_trials-D1_correct, D2_trials-D2_correct) 
        
        perf[0, i_day] = ND_correct/ND_trials*100 
        perf[1, i_day] = D1_correct/D1_trials*100 
        perf[2, i_day] = D2_correct/D2_trials*100 
    
    ax = fig.add_subplot('13'+str(i_mouse+1)) 
    
    days = np.arange(1, len(gv.sessions)+1 ) 
    plt.plot(gv.days, perf[0], color=pal[0]) 
    plt.plot(gv.days, perf[1], color=pal[1]) 
    plt.plot(gv.days, perf[2], color=pal[2])
    ax.set_title(gv.mouse) 
    plt.xlabel('Day') 
    plt.ylabel('Performance') 
    plt.ylim([40,100]) 
    plt.xlim([gv.days[0], gv.days[-1]]) 
    plt.xticks(gv.days) 
    
pl.figDir()
gv.IF_SAVE=1
pl.save_fig(figtitle) 
    
