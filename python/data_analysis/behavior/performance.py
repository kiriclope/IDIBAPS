#!/usr/bin/env python3

import click
import numpy as np 
import matplotlib.pyplot as plt 

import data.constants as gv 
import data.plotting as pl 
import data.utils as data 
import data.fct_facilities as fac 
fac.SetPlotParams() 

@click.command()
@click.argument('n_days') 
@click.argument('trial') 
@click.argument('event')

def main(n_days, trial, event):
    gv.n_days = int(n_days)
    
    print(trial, event)
    pal = ['r','b','y'] 
    label = [0, 13, 14]
    gv.data_type = 'raw' 
    
    # Col 1: Sample Odor (17,S1; 18,S2)
    # Col 2: Test Odor (11,T1; 12,T2)
    # Col 3: Response in DPA Task (1,Hit; 2,Miss; 3,FalseAlarm; 4,Correct Rejection) 
    # Col 5: Distractor Odor (0,No Distractor; 13,D1, Go-Trial; 14,D2, NoGo-Trial)
    # Col 9: Laser (0,Laser Off; 1,Laser On)
    
    figtitle = trial + '_' + event
    fig = plt.figure(figtitle, figsize=(2.1*1.25*3, 1.85*1.25)) 

    for i_mouse, gv.mouse in enumerate(gv.mice[1:4]): 
        
        ax = fig.add_subplot('13'+str(i_mouse+1))         
        ax.set_title(gv.mouse)
        data.get_days()
        
        trial_type = np.empty( ( len(gv.trials), len(gv.days) ) ) 
        
        mean_perf = np.empty( ( len(gv.trials), len(gv.days) ) ) 
        std_perf = np.empty( ( len(gv.trials), len(gv.days) ) ) 
        lower_perf = np.empty( ( len(gv.trials), len(gv.days) ) ) 
        upper_perf = np.empty( ( len(gv.trials), len(gv.days) ) ) 
       
        for i_day, gv.day in enumerate(gv.days): 
            X, y_labels = data.get_fluo_data() 
            
            bool_pair = ( (y_labels[0]==17) & (y_labels[1]==11) ) | ( (y_labels[0]==18) & (y_labels[1]==12) ) 
            bool_unpair = ( (y_labels[0]==17) & (y_labels[1]==12) ) | ( (y_labels[0]==18) & (y_labels[1]==11) ) 
            
            bool_hit = y_labels[2]==1
            bool_miss = y_labels[2]==2 
            bool_false = y_labels[2]==3 
            bool_correct = y_labels[2]==4
            
            if trial=='paired':
                if event=='hit': 
                    bool_type = bool_hit
                    plt.ylabel('Paired trials, Hit')
                elif event=='miss': 
                    bool_type = bool_miss 
                    plt.ylabel('Paired trials, Miss')
                elif event=='false': 
                    bool_type = bool_false
                    plt.ylabel('Paired trials, FA')
                elif event=='correct': 
                    bool_type = bool_correct
                    plt.ylabel('Paired trials, CR')
                    
            elif trial=='unpaired':
                if event=='hit': 
                    bool_type = bool_hit
                    plt.ylabel('Unpaired trials, Hit')
                elif event=='miss': 
                    bool_type = bool_miss 
                    plt.ylabel('Unpaired trials, Miss')
                elif event=='false': 
                    bool_type = bool_false
                    plt.ylabel('Unpaired trials, FA')
                elif event=='correct': 
                    bool_type = bool_correct
                    plt.ylabel('Unpaired trials, CR')
                    
            elif trial=='all':
                if event=='hit': 
                    bool_type = bool_hit
                    plt.ylabel('All trials, Hit')
                elif event=='miss': 
                    bool_type = bool_miss 
                    plt.ylabel('All trials, Miss')
                elif event=='false': 
                    bool_type = bool_false
                    plt.ylabel('All trials, FA')
                elif event=='correct': 
                    bool_type = bool_correct
                    plt.ylabel('All trials, CR')
                elif event=='hit_correct':
                    bool_type = bool_hit | bool_correct
                    plt.ylabel('All trials, performance')
                    
            for i_trial in range(len(gv.trials)): 
            
                bool_trial = (y_labels[4]==label[i_trial]) & (y_labels[8]==0)
                
                if trial =='paired':
                    bool_trial = bool_trial & bool_pair
                elif trial =='unpaired':
                    bool_trial = bool_trial & bool_unpair 
                    
                trial_type[i_trial, i_day] = np.count_nonzero(bool_trial) 
                task = np.where( bool_trial )[0] 
                print('trial', gv.trials[i_trial], task.shape) 
            
                perf_sample = np.empty(1000) 
                for i_sample in range(1000):
                    idx_sample = np.random.randint(0, task.shape[0], task.shape[0])
                
                    bool_sample = bool_type[task[idx_sample]] 
                
                    perf_sample[i_sample] = np.count_nonzero(bool_sample) / task.shape[0]*100 
                
                    # print('perf', perf_sample[i_sample]) 
            
                mean_perf[i_trial, i_day] = np.mean(perf_sample) 
                std_perf[i_trial, i_day] = np.std(perf_sample)
                
                lower_perf[i_trial, i_day] = mean_perf[i_trial, i_day] - np.percentile(perf_sample, 25, axis=-1) 
                upper_perf[i_trial, i_day] = np.percentile(perf_sample, 75, axis=-1) - mean_perf[i_trial, i_day] 
                
                print('trial', gv.trials[i_trial], 'day', gv.days[i_day],
                      'mean perf', mean_perf[i_trial, i_day], 'std', std_perf[i_trial, i_day])
        
        for i_trial in range(len(gv.trials)):
            error = np.absolute( np.vstack([ lower_perf[i_trial], upper_perf[i_trial] ] ) )
            
            plt.errorbar(gv.days, mean_perf[i_trial], yerr=error,
                         color=gv.pal[i_trial], fmt='-o', markersize=2.5, elinewidth=0.5, capsize=3) 
            plt.xlabel('Day') 
            plt.xlim([gv.days[0]-1, gv.days[-1]+1]) 
            plt.xticks(gv.days) 
            plt.ylim([-10,110]) 
            
    pl.figDir()
    gv.IF_SAVE=1
    pl.save_fig(figtitle) 
    plt.close('all')
        
if __name__ == '__main__': 
    main()
