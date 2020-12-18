import numpy as np
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
