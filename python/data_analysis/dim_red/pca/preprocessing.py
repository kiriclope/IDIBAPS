import numpy as np 
from sklearn.preprocessing import StandardScaler 
import scipy.stats as st 

def center(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=False)
    Xc = ss.fit_transform(X.T).T
    return Xc

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz

def conf_inter(y):
    ci = []
    for i in range(y.shape[0]):
        ci.append( st.t.interval(0.95, y.shape[1]-1, loc=np.mean(y[i,:]), scale=st.sem(y[i,:])) )
    ci = np.array(ci).T

    return ci
