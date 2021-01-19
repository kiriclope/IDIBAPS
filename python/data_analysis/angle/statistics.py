import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

def t_test(x,y,alternative='both-sided'): 
    _, double_p = stats.ttest_ind(x,y,equal_var = False) 
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval

def get_p_values(cos_sample):
    ''' N_trials, N_epochs, N_boots '''
    p_values = np.empty( ( cos_sample.shape[0]-1, cos_sample.shape[1]-1) ) 
    for n_trial in range(1, cos_sample.shape[0]): # trials D1, D2
        for n_epoch in range(1, cos_sample.shape[1]): # epochs MD, LD 
            sample_1  = cos_sample[0, n_epoch] # N_boots, trial ND epoch MD, LD 
            sample_2  = cos_sample[n_trial, n_epoch] # trial D1, D2, epoch MD, LD => ND/MD vs D1/MD, ND/LD vs D1/LD, ND/
            p_values[n_trial-1, n_epoch-1] = t_test(sample_2, sample_1, alternative='both-sided') 
            # note sample_2 then sample_1 for H0: S2>=S1, Ha S1>S2 
    return p_values

def add_pvalue(p_values): 
    cols = 0.25*np.arange(p_values.shape[0]+1) 
    high = [1.25, 1.1] 

    for n_cols in range(1, len(cols)): 
        for n_epoch in range(p_values.shape[1]): 
            plt.plot( [n_epoch + cols[0], n_epoch + cols[n_cols]] , [high[n_cols-1], high[n_cols-1]] , lw=.8, c='k') 
            
            if p_values[n_cols-1,n_epoch]<=.001: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "***", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.01: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "**", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]<=.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1]-.05, "*", ha='center', va='bottom', color='k', fontsize=8) 
            elif p_values[n_cols-1,n_epoch]>.05: 
                plt.text((2*n_epoch+cols[0]+cols[n_cols])*.5, high[n_cols-1], "ns", ha='center', va='bottom', color='k', fontsize=6) 
