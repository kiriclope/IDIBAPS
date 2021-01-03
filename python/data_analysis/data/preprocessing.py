from .libs import * 
from . import constants as gv
from . import progressbar as pg
from . import featureSel as fs

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from oasis.functions import deconvolve 

from joblib import Parallel, delayed, parallel_backend

import scipy.signal

def center(X):
    scaler = StandardScaler(with_mean=True, with_std=False)
    Xc = scaler.fit_transform(X.T).T
    return Xc

def z_score(X): 
    ss = StandardScaler() 
    ss.fit(X[:,gv.bins_BL].T) 
    Xz = ss.transform(X.T).T 
    # Xz = ss.fit_transform(X.T).T 
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

def dFF0(X, AVG_TRIALS=0):
    if not AVG_TRIALS:
        F0 = np.mean(X[:,:,gv.bins_BL],axis=2) 
        F0 = F0[:,:, np.newaxis] 
    else: 
        F0 = np.mean( np.mean(X[:,:,gv.bins_BL],axis=2), axis=0) 
        F0 = F0[np.newaxis,:, np.newaxis] 
    return (X-F0) / (F0 + gv.eps) 

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
        # filter at.
        cutoff = min(1.0, fw_base / nyq_rate)

        # Make a set of weights to use with our taps.
        # We use an FIR filter with a Hamming window.
        b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')

        # The default padlen for filtfilt is 3 * nfilt, but in case our
        # dataset is small, we need to make sure padlen is not too big
        padlen = min(3 * nfilt, rawF.shape[axis] - 1)

        # Use filtfilt to filter with the FIR filter, both forwards and
        # backwards.
        filtered_f = scipy.signal.filtfilt(b, [1.0], rawF, axis=axis,
                                           padlen=padlen)

    # Take a percentile of the filtered signal
    baselineF0 = np.percentile(filtered_f, base_pctle, axis=axis,
                               keepdims=keepdims)

    baselineF0 = baselineF0.T
    baselineF0 = baselineF0[:,np.newaxis,:]
    return baselineF0


def bin_data(data, bin_step, bin_size):
    # bin_step number of pts btw bins, bin_size number of size in each bin
    bin_array = [np.mean(np.take(data,np.arange(int(i*bin_step),int(i*bin_step+bin_size)), axis=2), axis=2) for i in np.arange(data.shape[2]//bin_step-1)]
    bin_array = np.array(bin_array)
    bin_array = np.rollaxis(bin_array,0,3)
    return bin_array

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
        X_ED = np.mean(X[:,:,0:len(gv.bins_ED)],axis=-1) 
        X_MD = np.mean(X[:,:,len(gv.bins_ED):len(gv.bins_ED)+len(gv.bins_MD)],axis=-1) 
        X_LD = np.mean(X[:,:,len(gv.bins_ED)+len(gv.bins_MD):len(gv.bins_ED)+len(gv.bins_MD)+len(gv.bins_LD)],axis=-1) 
        X_STIM = X_ED 
        
    elif gv.trialsXepochs: 
        print(gv.trial,'avg trials x epochs') 
        X_STIM = np.hstack(X[:,:,gv.bins_STIM[:]-gv.bin_start]).T 
        X_ED = np.hstack(X[:,:,gv.bins_ED[:]-gv.bin_start]).T 
        X_MD = np.hstack(X[:,:,gv.bins_MD[:]-gv.bin_start]).T 
        X_LD = np.hstack(X[:,:,gv.bins_LD[:]-gv.bin_start]).T 
    else: 
        print(gv.trial,'avg over epochs')
        if not gv.EDvsLD:
            X_STIM = np.mean(X[:,:,gv.bins_STIM[:]-gv.bin_start],axis=2)
            
        X_ED = np.mean(X[:,:,gv.bins_ED[:]-gv.bin_start],axis=2) 
        X_MD = np.mean(X[:,:,gv.bins_MD[:]-gv.bin_start],axis=2) 
        X_LD = np.mean(X[:,:,gv.bins_LD[:]-gv.bin_start],axis=2) 
        
    if gv.FEATURE_SELECTION: 
        # idx = fs.featSel.var_fit_transform(X_ED, threshold) 
        # X_ED = np.delete(X_ED, idx, axis=1) 
        
        X_ED = fs.featSel.select_best(X_ED, y) 
        X_MD = fs.featSel.select_best(X_MD, y) 
        X_LD = fs.featSel.select_best(X_LD, y) 
        
        print(X_ED.shape, X_MD.shape, X_LD.shape) 
    
    if len(gv.epochs)==3: 
        X_epochs = np.empty( (3, X_ED.shape[0], np.amax([X_ED.shape[1], X_MD.shape[1], X_LD.shape[1]]) ) ) 
        X_epochs[0,:,0:X_ED.shape[1]] = X_ED 
        X_epochs[1,:,0:X_MD.shape[1]] = X_MD 
        X_epochs[2,:,0:X_LD.shape[1]] = X_LD 
    else: 
        X_epochs = np.empty( (4, X_ED.shape[0], np.amax([X_STIM.shape[1], X_ED.shape[1], X_MD.shape[1], X_LD.shape[1]]) ) ) 
        X_epochs[0,:,0:X_STIM.shape[1]] = X_STIM 
        X_epochs[1,:,0:X_ED.shape[1]] = X_ED 
        X_epochs[2,:,0:X_MD.shape[1]] = X_MD 
        X_epochs[3,:,0:X_LD.shape[1]] = X_LD 
        
    X_epochs = np.moveaxis(X_epochs,0,2) 
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

# def deconvolve_X(X):

#     F0 = np.mean(X[:,gv.bins_BL],axis=1) 
#     with pg.tqdm_joblib(pg.tqdm(desc='deconvolve', total=X.shape[0])) as progress_bar: 
#         result = Parallel(n_jobs=gv.num_cores)(delayed(deconvolve)(X[n_neuron], penalty=1, b=F0) for n_neuron in range(X.shape[0]) )

#     result = np.array(result).T
#     # print(result.shape)
#     X_deconv = (np.stack(result[0])-F0[:, np.newaxis])/(F0[:, np.newaxis] + gv.eps) 
#     # print(X_deconv.shape)
    
#     # F0 = np.mean(X[:,:,gv.bins_BL],axis=2)     
#     # with pg.tqdm_joblib(pg.tqdm(desc='deconvolve', total=int( X.shape[0] * X.shape[1] ) )) as progress_bar: 
#     #     result = Parallel(n_jobs=gv.num_cores)(delayed(deconvolve)(X[trial, n_neuron], penalty=1, b=F0[trial, n_neuron] ) for trial in range(X.shape[0]) for n_neuron in range(X.shape[1]) ) 

#     # result = np.array(result).T
#     # result = np.array(result).reshape( (5, X.shape[0], X.shape[1]) )
#     # result = np.moveaxis(result, 1, 2)
#     # # print(result.shape)
    
#     # X_deconv = np.stack( np.hstack(result[0]) ).reshape( (X.shape[0], X.shape[1], X.shape[2]) ) 
#     # # print(X_denoised.shape)

#     # X_spikes = np.stack( np.hstack(result[1]) ).reshape( (X.shape[0], X.shape[1], X.shape[2]) ) 
#     # print(X_denoised.shape)
    
#     # X_denoised = np.empty( (X.shape[0], X.shape[1]) )
#     # X_spikes = np.empty( (X.shape[0], X.shape[1]) ) 

#     # for n_neuron in range(X.shape[0]):
#     #     X_denoised[n_neuron], X_spikes[n_neuron], F0, g, lam = deconvolve(X[n_neuron], penalty=1) 

#     return X_deconv

# def remove_outliers(X):
#     ''' for each trial remove outlier neurons ''' 
#     q25, q75 = np.percentile(X, 25, axis=-1), np.percentile(X, 75, axis=-1) 
#     iqr = q75 - q25
    
#     cut_off = iqr * 1.5 
#     lower, upper = q25 - cut_off, q75 + cut_off 
    
#     outliers = [neuron for neuron in X[] if x < lower or x > upper]
#     outliers_removed = [x for x in X if x > lower and x < upper]
