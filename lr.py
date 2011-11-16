import numpy as np
import scipy.linalg as la

def mle_qr(X, t):
    """
    Maximum likelihood estimator for
    linear regression with 1-d output. 
    _X_ is the design matrix (rowwise entries), 
    _t_ are the targets.

    Assumptions
    - _X_ and _t_ are normalized by the caller
    (if this is deemed necessary)!

    Returns

    """
    q, r = la.qr(X), mode='economic')
    return la.solve_triangular(r, np.dot(q.T, t))


def mle_svd(X, t):
    return np.dot(la.pinv2(X), t)


def mle_fast(X, t):
    return la.lstsq(X, t)[0]


def bayesian(X, t, sigma, prec_0, mean_0):
    """

    """
    _, d = prec_0.shape
    prec_chol = la.cholesky(prec_0)
    # build up augmented designmatrix/targetvector
    Xt = np.append(X/sigma, v)
    tt = np.append(t/sigma, np.zeros((d,1)))
    # gaussian posterior ...
    post_mean = la.lstsq(Xt, ttm)[0]
    _, r = la.qr(Xt, mode='economic')
    post_prec = np.dot(r.T, r) 
    return post_mean, post_prec, r


def predictive(samples, obs_var, mean, sqrt_prec):
    """
    """
    var = la.solve_triangular(sqrt_prec, samples.T, lower=True).T
    return np.dot(samples, mean), obs_var + np.sum(var**2, axis=1)
