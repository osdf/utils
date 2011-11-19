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
    q, r = la.qr(X, mode='economic')
    return la.solve_triangular(r, np.dot(q.T, t))


def mle_svd(X, t):
    return np.dot(la.pinv2(X), t)


def mle_fast(X, t):
    return la.lstsq(X, t)[0]


def bayesian(X, t, sigma, mean_0, prec_0):
    """

    """
    prec_chol = la.cholesky(prec_0)
    # build up augmented designmatrix/targetvector
    Xt = np.append(X/sigma, prec_chol, axis=0)
    tt = np.append(t/sigma, np.dot(prec_chol, mean_0), axis=0)
    # gaussian posterior ...
    post_mean = la.lstsq(Xt, tt)[0]
    _, r = la.qr(Xt, mode='economic')
    post_prec = np.dot(r.T, r) 
    return post_mean, post_prec, r


def predictive(samples, sigma, mean, sqrt_prec):
    """
    """
    var = la.solve_triangular(sqrt_prec, samples.T, trans=1).T
    return np.dot(samples, mean), sigma + np.sum(var**2, axis=1) 


def robust(X, t, mean_0, prec_0, a, b):
    """
    """
    prec_chol = la.cholesky(prec_0)
    Xt = np.append(X, prec_chol, axis=0)
    tt = np.append(t, np.dot(prec_chol, mean_0), axis=0)
    n, d = Xt.shape
    w = np.ones(n,1)
    w_new= np.ones(n-d,1)
    while True:
        w[:-d] *= w_new/w[:-d]
        mean = la.lstsq(np.sqrt(w/var)*Xt, np.sqrt(w/var)*tt)
        _, r = la.qr(np.sqrt(w/var)*Xt)
    return


def test(nos, nob):
    samples = np.random.rand(nos, 1)
    centers = np.linspace(0, 1, nob)
    width = 4./10
    X = np.zeros((nos, nob+1))
    for i, s in enumerate(samples):
        X[i, 0] = 1
        for j, c in enumerate(centers):
            X[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)
    y = np.sin(2*np.pi*samples) + 0.1*np.random.randn(nos, 1)
    m_0 = np.zeros((nob+1, 1))
    p_0 = 0.0001 * np.eye(nob+1)
    m, p, r = bayesian(X, y, 0.1, m_0, p_0)
    tests = np.linspace(0, 1, 100)
    Xt = np.zeros((tests.shape[0], nob+1))
    for i, s in enumerate(tests):
        Xt[i, 0] = 1
        for j, c in enumerate(centers):
            Xt[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)
    pred, var = predictive(Xt, 0.1, m, r)
    import pylab
    pylab.plot(samples, y, 'ro')
    pylab.plot(tests, pred, 'g')
    pylab.fill_between(tests, pred.flatten() + np.sqrt(var), pred.flatten() - np.sqrt(var), alpha=0.1, color='g')
    pylab.plot(tests, np.sin(2*np.pi*tests), 'r', lw=2)
