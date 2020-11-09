from .libs import * 
from . import constants as gv

import scipy.signal

def center(X):
    # X: ndarray, shape (n_features, n_samples)
    scaler = StandardScaler(with_mean=True, with_std=False)
    Xc = scaler.fit_transform(X.T).T
    return Xc

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
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
