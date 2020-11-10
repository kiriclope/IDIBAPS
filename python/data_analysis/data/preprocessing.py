from .libs import * 
from . import constants as gv

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import scipy.signal

def center(X):
    # X: ndarray, shape (n_features, n_samples)
    scaler = StandardScaler(with_mean=True, with_std=False)
    Xc = scaler.fit_transform(X.T).T
    return Xc

def z_score(X): 
    # X: ndarray, shape (n_features, n_samples) 
    ss = StandardScaler(with_mean=True, with_std=True) 
    # ss.fit(X[:,gv.bins_BL].T) 
    # Xz = ss.transform(X.T).T 
    Xz = ss.fit_transform(X.T).T 
    return Xz 

def normalize(X):
    # X: ndarray, shape (n_features, n_samples)
    Xmin = np.amin(X, axis=1)
    Xmax = np.amax(X, axis=1)
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