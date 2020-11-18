import numpy as np

def decision(coefs, X, intercept):
    return np.dot(X, coefs) + intercept

def get_se(X, y, clf):
    """StdErr per variable estimation.
    https://en.wikipedia.org/wiki/Ordinary_least_squares 
    """
    MSE = np.mean((y - clf.predict(X).T)**2) 
    # numerically unstable below with openblas if rcond is less than that
    var_est = MSE * np.diag(np.linalg.pinv(np.dot(X.T, X), rcond=1e-10))
    SE_est = np.sqrt(var_est)
    return SE_est

def get_probs(clf, X, SE_est, z=1.96):
    """Estimate CI given data, StdErrors and model."""
    coefs = np.ravel(clf.coef_)
    upper = coefs + (z * SE_est)
    lower = coefs - (z * SE_est)

    prob = 1. / (1. + np.exp(-decision(coefs, X, clf.intercept_)))
    upper_prob = 1. / (1. + np.exp(-decision(upper, X, clf.intercept_)))
    lower_prob = 1. / (1. + np.exp(-decision(lower, X, clf.intercept_)))

    stacked = np.vstack((lower_prob, upper_prob))
    up = np.max(stacked, axis=0)
    lo = np.min(stacked, axis=0)
    return prob, up, lo
