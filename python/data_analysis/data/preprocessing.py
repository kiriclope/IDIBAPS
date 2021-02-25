from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_regression, mutual_info_classif, f_classif

from joblib import Parallel, delayed, parallel_backend
from meegkit.detrend import detrend
from oasis.functions import deconvolve

from .libs import * 
from . import constants as gv
from . import progressbar as pg
from . import featureSel as fs

def center(X):
    scaler = StandardScaler(with_mean=True, with_std=False)
    Xc = scaler.fit_transform(X.T).T
    return Xc

def z_score(X): 
    scaler = StandardScaler()
    
    if X.ndim>2:
        Xz = X
        for i in range(X.shape[0]):
            Xt = X[i]
            if gv.Z_SCORE:
                Xz[i] = scaler.fit_transform(Xt.T).T
            elif gv.Z_SCORE_BL : 
                scaler.fit(Xt[...,gv.bins_BL].T) 
                Xz[i] = scaler.transform(Xt.T).T 
            
    else:
            
        if gv.Z_SCORE:
            Xz = scaler.fit_transform(X.T).T
        elif gv.Z_SCORE_BL : 
            scaler.fit(X[...,gv.bins_BL].T) 
            Xz = scaler.transform(X.T).T
            
    return Xz 

def normalize(X):
    Xmin = np.amin(X[:, gv.bins_ED], axis=1) 
    Xmax = np.amax(X[:, gv.bins_ED], axis=1) 
    
    Xmin = Xmin[:,np.newaxis] 
    Xmax = Xmax[:,np.newaxis]
    
    return (X-Xmin)/(Xmax-Xmin+gv.eps) 

def conf_inter(y): 
    ci = []
    for i in range(y.shape[0]):
        ci.append( stats.t.interval(0.95, y.shape[1]-1, loc=np.mean(y[i,:]), scale=stats.sem(y[i,:])) )
    ci = np.array(ci).T

    return ci

def dFF0_remove_silent(X): 
    ''' N_trials, N_neurons, N_times '''
    if gv.AVG_F0_TRIALS: 
        F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis]
    else:
        F0 = np.mean(X[...,gv.bins_BL],axis=-1) 
        F0 = F0[..., np.newaxis]

    if gv.F0_THRESHOLD is not None: 
        # removing silent neurons 
        idx = np.argwhere(F0<=gv.F0_THRESHOLD) 
        F0 = np.delete(F0, idx, axis=-2) 
        X = np.delete(X, idx, axis=-2)
        
    return (X-F0) / (F0 + gv.eps) 
    
def dFF0(X): 
    if not gv.AVG_F0_TRIALS: 
        F0 = np.mean(X[...,gv.bins_BL],axis=-1)        
        # F0 = np.percentile(X, 15, axis=-1) 
        F0 = F0[..., np.newaxis] 
    else: 
        F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis]
        
    return (X-F0) / (F0 + gv.eps) 

def dF(X): 
    if not gv.AVG_F0_TRIALS: 
        F0 = np.mean(X[...,gv.bins_BL],axis=-1)        
        # F0 = np.percentile(X, 15, axis=-1) 
        F0 = F0[..., np.newaxis] 
    else: 
        F0 = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis]
        
    return (X-F0)

def findBaselineF0(rawF, fs, axis=0, keepdims=False): 
    """Find the baseline for a fluorescence imaging trace line.
    
    The baseline, F0, is the 5th-percentile of the 1Hz
    lowpass filtered signal.
    
    Parameters
    ----------
    rawF : array_like
        Raw fluorescence signal.
    fs : float
        Sampling frequency of rawF, in Hz.
    axis : int, optional
        Dimension which contains the time series. Default is 0.
    keepdims : bool, optional
        Whether to preserve the dimensionality of the input. Default is
        `False`.
    
    Returns
    -------
    baselineF0 : numpy.ndarray
        The baseline fluorescence of each recording, as an array.
    
    Note
    ----
    In typical usage, the input rawF is expected to be sized
    `(numROI, numTimePoints, numRecs)`
    and the output will then be sized `(numROI, 1, numRecs)`
    if `keepdims` is `True`.
    """
    
    rawF = np.moveaxis(rawF.T,0,1)
    print('#neurons x #time x #trials', rawF.shape)
    
    # Parameters --------------------------------------------------------------
    nfilt = 30  # Number of taps to use in FIR filter
    fw_base = 1  # Cut-off frequency for lowpass filter, in Hz
    base_pctle = 5  # Percentile to take as baseline value
    
    # Main --------------------------------------------------------------------
    # Ensure array_like input is a numpy.ndarray
    rawF = np.asarray(rawF)
    
    # Remove the first datapoint, because it can be an erroneous sample
    rawF = np.split(rawF, [1], axis)[1]
    
    if fs <= fw_base:
        # If our sampling frequency is less than our goal with the smoothing
        # (sampling at less than 1Hz) we don't need to apply the filter.
        filtered_f = rawF
        
    else:
        # The Nyquist rate of the signal is half the sampling frequency
        nyq_rate = fs / 2.0
        
        # Cut-off needs to be relative to the nyquist rate. For sampling
        # frequencies in the range from our target lowpass filter, to
        # twice our target (i.e. the 1Hz to 2Hz range) we instead filter
        # at the Nyquist rate, which is the highest possible frequency to
        # filter 
        cutoff = min(1.0, fw_base / nyq_rate) 
        
        # Make a set of weights to use with our taps.
        # We use an FIR filter with a Hamming window.
        b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')
        
        # The default padlen for filtfilt is 3 * nfilt, but in case our
        # dataset is small, we need to make sure padlen is not too big
        padlen = min(3 * nfilt, rawF.shape[axis] - 1)
        
        # Use filtfilt to filter with the FIR filter, both forwards and
        # backwards. 
        filtered_f = scipy.signal.filtfilt(b, [1.0], rawF, axis=axis, padlen=padlen) 
        
    # Take a percentile of the filtered signal
    baselineF0 = np.percentile(filtered_f, base_pctle, axis=axis, keepdims=keepdims)

    baselineF0 = baselineF0.T
    baselineF0 = baselineF0[:,np.newaxis,:]
    return baselineF0


def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array

def detrend_loop(X, trial, neuron, order):
    X_det, _, _ = detrend(X[trial, neuron], order)
    return X_det

def detrend_X(X, order=3):
    with pg.tqdm_joblib(pg.tqdm(desc='trial ' + gv.trial +' detrend X', total=int(X.shape[0]*X.shape[1]) ) ) as progress_bar: 
        dum = Parallel(n_jobs=gv.num_cores)(delayed(detrend_loop)(X, trial, neuron, order) 
                                            for trial in range(X.shape[0]) 
                                            for neuron in range(X.shape[1]) )
               
        X = np.asarray(dum).reshape(X.shape[0], X.shape[1], X.shape[2])
    return X

def detrend_data(X_trial, poly_fit=1, degree=7): 
    """ Detrending of the data, if poly_fit=1 uses polynomial fit else linear fit. """
    # X_trial : # neurons, # times 
    
    model = LinearRegression()
    fit_values_trial = []

    indexes = range(0, X_trial.shape[1]) # neuron index 
    values = np.mean(X_trial,axis=0) # mean fluo value 
    
    indexes = np.reshape(indexes, (len(indexes), 1))
    
    if poly_fit:
        poly = PolynomialFeatures(degree=degree) 
        indexes = poly.fit_transform(indexes) 
            
    model.fit(indexes, values)
    fit_values = model.predict(indexes) 
    fit_values_trial = np.array(fit_values)
    
    # for i in range(0, X_trial.shape[0]): # neurons 
    #     indexes = range(0, X_trial.shape[1]) # neuron index 
    #     values = X_trial[i] # fluo value 
                
    #     indexes = np.reshape(indexes, (len(indexes), 1))

    #     if poly_fit:
    #         poly = PolynomialFeatures(degree=degree) 
    #         indexes = poly.fit_transform(indexes) 

    #     model.fit(indexes, values)
    #     fit_values = model.predict(indexes) 
        
    #     fit_values_trial.append(fit_values) 
        
    # fit_values_trial = np.array(fit_values_trial)
    return fit_values_trial

def feature_selection(method='variance'):

    X_avg = np.mean(X[:,:,gv.bins_ED_MD_LD],axis=-1) 

    if 'variance' in method :
        idx = fs.featSel.var_fit_transform(X_avg, threshold=threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1)
            
    if 'mutual' in method:
        idx = fs.featSel.select_best(X_avg, y, percentage=1-threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1)
            
    if 'correlation' in method:
        idx = fs.featSel.select_indep(X_avg, threshold=threshold) 
        X_avg = np.delete(X_avg, idx, axis=1) 
        X = np.delete(X, idx, axis=1) 

def avg_epochs(X, y=None, threshold=.1): 
    
    if gv.ED_MD_LD: 
        X_ED = np.mean(X[...,0:len(gv.bins_ED)],axis=-1) 
        X_MD = np.mean(X[...,len(gv.bins_ED):len(gv.bins_ED)+len(gv.bins_MD)],axis=-1) 
        X_LD = np.mean(X[...,len(gv.bins_ED)+len(gv.bins_MD):len(gv.bins_ED)+len(gv.bins_MD)+len(gv.bins_LD)],axis=-1) 
        X_STIM = X_ED 
        
    else: 
        print('average time bins over epochs:', gv.epochs)
        if not gv.EDvsLD:
            X_STIM = np.mean(X[...,gv.bins_STIM[:]-gv.bin_start],axis=2)
            
        X_ED = np.mean(X[...,gv.bins_ED[:]-gv.bin_start],axis=-1) 
        X_MD = np.mean(X[...,gv.bins_MD[:]-gv.bin_start],axis=-1) 
        X_LD = np.mean(X[...,gv.bins_LD[:]-gv.bin_start],axis=-1) 
        
        if gv.trialsXepochs or gv.CONCAT_BINS: 
            print(gv.trial,'concatenate bins and average') 
            # X_STIM = np.hstack(X[...,gv.bins_STIM[:]-gv.bin_start]).T
                
            X_ED = np.hstack(X[...,gv.bins_ED[:]-gv.bin_start].T).T 
            X_MD = np.hstack(X[...,gv.bins_MD[:]-gv.bin_start].T).T 
            X_LD = np.hstack(X[...,gv.bins_LD[:]-gv.bin_start].T).T 
            print(X_ED.shape, X_MD.shape) 
            
    if gv.FEATURE_SELECTION: 
        # idx = fs.featSel.var_fit_transform(X_ED, threshold) 
        # X_ED = np.delete(X_ED, idx, axis=1) 
        
        X_ED = fs.featSel.select_best(X_ED, y) 
        X_MD = fs.featSel.select_best(X_MD, y) 
        X_LD = fs.featSel.select_best(X_LD, y) 
        
        print(X_ED.shape, X_MD.shape, X_LD.shape) 
    
    if len(gv.epochs)==3: 
        X_epochs = np.empty( tuple([3])+ X_ED.shape ) 
        # print('X', X_epochs.shape, 'X_ED', X_ED.shape)
        X_epochs[0] = X_ED 
        X_epochs[1] = X_MD 
        X_epochs[2] = X_LD 
    else: 
        X_epochs = np.empty( tuple([4])+ X.shape[:-1] ) 
        X_epochs[0] = X_STIM 
        X_epochs[1] = X_ED 
        X_epochs[2] = X_MD 
        X_epochs[3] = X_LD 

    X_epochs = np.moveaxis(X_epochs,0,-1)  

    return X_epochs 

def selectiveNeurons(X_S1, X_S2, Threshold=.01):
    X_S1n = X_S1 
    X_S2n = X_S2 
        
    for i in range(X_S1n.shape[0]): 
        # X_S1n[i] = normalize(X_S1n[i]) 
        # X_S2n[i] = normalize(X_S2n[i]) 
        X_S1n[i] = z_score(X_S1n[i]) 
        X_S2n[i] = z_score(X_S2n[i]) 
        
    sel_idx = (X_S1n - X_S2n)/(X_S1n + X_S2n + gv.eps) 
    
    if gv.ED_MD_LD: 
        sel_idx = np.mean(sel_idx, axis=-1) 
    else:
        sel_idx = np.mean(sel_idx[:,:,gv.bins_STIM-gv.bin_start], axis=-1) 
        
    idx = np.where(abs(sel_idx)<=Threshold) 
    X_S1 = np.delete(X_S1, idx, axis=1) 
    X_S2 = np.delete(X_S2, idx, axis=1) 
    
    return X_S1, X_S2, idx

def deconvolveFluo(X):

    # F0 = np.empty( (X.shape[0], X.shape[1]) ) 
    # F0[:] = np.mean( np.mean(X[...,gv.bins_BL],axis=-1), axis=0 ) 
    F0 = np.mean(X[...,gv.bins_BL],axis=-1) 
    # F0 = np.percentile(X, 15, axis=-1) 
    
    # def F0_loop(X, n_trial, n_neuron, bins): 
    #     X_ij = X[n_trial, n_neuron]        
    #     c, s, b, g, lam = deconvolve(X_ij, penalty=1) 
    #     return b
    
    # # loop over trials and neurons 
    # with pg.tqdm_joblib(pg.tqdm(desc='F0', total=X.shape[0]*X.shape[1])) as progress_bar: 
    #     F0 = Parallel(n_jobs=gv.num_cores)(delayed(F0_loop)(X, n_trial, n_neuron, gv.bins_BL) 
    #                                         for n_trial in range(X.shape[0]) 
    #                                         for n_neuron in range(X.shape[1]) )
        
    # F0 = np.array(F0).reshape( (X.shape[0], X.shape[1]) ) 
            
    def X_loop(X, F0, n_trial, n_neuron):
        X_ij = X[n_trial, n_neuron]
        F0_ij = F0[n_trial, n_neuron]
        c, s, b, g, lam = deconvolve(X_ij, penalty=1, b=F0_ij) 
        return c 
    
    def S_loop(X, F0, n_trial, n_neuron):
        X_ij = X[n_trial, n_neuron]
        F0_ij = F0[n_trial, n_neuron]
        c, s, b, g, lam = deconvolve(X_ij, penalty=1, b=F0_ij)
        return s 
    
    # # loop over trials and neurons 
    # with pg.tqdm_joblib(pg.tqdm(desc='denoise', total=X.shape[0]*X.shape[1])) as progress_bar: 
    #     X_dcv = Parallel(n_jobs=gv.num_cores)(delayed(X_loop)(X, F0, n_trial, n_neuron) 
    #                                         for n_trial in range(X.shape[0]) 
    #                                         for n_neuron in range(X.shape[1]) ) 
    # X_dcv = np.array(X_dcv).reshape(X.shape) 
    
    with pg.tqdm_joblib(pg.tqdm(desc='deconvolve', total=X.shape[0]*X.shape[1])) as progress_bar: 
        S_dcv = Parallel(n_jobs=gv.num_cores)(delayed(S_loop)(X, F0, n_trial, n_neuron) 
                                              for n_trial in range(X.shape[0]) 
                                              for n_neuron in range(X.shape[1]) ) 
        
    S_dcv = np.array(S_dcv).reshape(X.shape)    
    # S_flt = savgol_filter(S_dcv, int(np.ceil(gv.frame_rate / 2.) * 2 + 1), polyorder = 5, deriv=0)
    
    def threshold_spikes(S_dcv, threshold): 
        S_dcv[S_dcv<=threshold] = 0 
        S_dcv[S_dcv>threshold] = 1 
        S_dcv = uniform_filter1d( S_dcv, int(gv.frame_rate/4) ) 
        return S_dcv 
    
    S_th = threshold_spikes(S_dcv, gv.DCV_THRESHOLD)    
    S_avg = np.mean(S_th[...,gv.bins_BL],axis=-1) 
    S_avg = S_avg[..., np.newaxis]

    print('X_avg', np.mean(S_avg))
    # removing silent neurons 
    # idx = np.argwhere(S_avg<=.001) 
    # S_th = np.delete(S_th, idx, axis=1)
    
    # print('X_dcv', S_th.shape[1]) 
    # gv.n_neurons = S_th.shape[1] 
    
    if gv.Z_SCORE | gv.Z_SCORE_BL: 
        
        if gv.Z_SCORE_BL: 
            gv.bins_z_score = gv.bins_BL 
        else: 
            gv.bins_z_score = gv.bins 
            
        def scaler_loop(S, n_trial, bins): 
            S_i = S[n_trial]
            scaler = StandardScaler() 
            scaler.fit(S_i[:,bins].T) 
            return scaler.transform(S_i.T).T 
        
        with pg.tqdm_joblib(pg.tqdm(desc='standardize', total=X.shape[0])) as progress_bar: 
            S_scaled = Parallel(n_jobs=gv.num_cores)(delayed(scaler_loop)(S_th, n_trial, gv.bins_z_score) 
                                                     for n_trial in range(X.shape[0]) ) 
            
        S_scaled = np.array(S_scaled) 
        
        return S_scaled 
        
    return S_th 

def soft_thresholding():    
    ''' see Diagnosis of multiple cancer types by shrunken centroids of gene expression, Tibshirani et al. , 2002, PNAS

    Xij, i features, j samples 

    Xik = sum_j_Ck Xij/nk, sum on j in class Ck

    Xi = sum_i Xij /n, sum on the n samples , mean over samples 

    dik = Xik - Xi / mk (si +s0) where si^2= 1/(n-K) sum_k sum_j_Ck (Xij -Xik)^2 pooled within class standard deviation 
                                       s0 = median(si), guard against the possibility of large dik  
                                       mk = sqrt(1/nk + 1/n), so that mk*si is the std of the numerator in dik
    We rewrite as

    Xik = Xi + mk (si+s0) dik 

    and shrink the dik with soft thresholding defined as: 
    d'ik = sign(dik) TL(|dik|-D) where TL is t->t if t>0 (strictly), else t->0 

    This method has the desirable property that many of the features are eliminated from the class prediction as 
    the shrinkage parameter, D,  is increased.

    
    '''
    
    return 0 

def prescreening(X, y, alpha=0.05, scoring=f_classif): 
    ''' X is trials x neurons 
    alpha is the level of significance 
    scoring is the statistics, use f_classif or mutual_info_classif 
    '''
    
    model = SelectKBest(score_func=scoring, k=X.shape[1])    
    model.fit(X,y) 
    pval = model.pvalues_.flatten() 
    non_sel = np.argwhere(pval>alpha) 
    X_sel = np.delete(X, non_selected, axis=1) 
    return X_sel 

def preprocess_X(X):
    
    if gv.F0_THRESHOLD is not None: 
        X = dFF0_remove_silent(X) 
        gv.n_neurons = X.shape[1]
        
    if gv.DECONVOLVE:
        X = deconvolveFluo(X) 
        
    else: 
        
        if gv.SAVGOL: 
            X = savgol_filter(X, int(np.ceil(gv.frame_rate / 2.) * 2 + 1), polyorder = 5, deriv=0, axis=-1) 
            
        if gv.Z_SCORE | gv.Z_SCORE_BL :
            print('z_score')
            X = z_score(X) 
            
    return X 
