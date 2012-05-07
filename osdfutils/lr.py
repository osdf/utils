"""

"""


import numpy as np
import scipy.linalg as la


import mvn


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


def bayesian(X, t, var, mean_0, prec_0, **params):
    """

    """
    result = {}
    prec_chol = la.cholesky(prec_0)
    # build up augmented designmatrix/targetvector
    Xt = np.append(X/np.sqrt(var), prec_chol, axis=0)
    tt = np.append(t/np.sqrt(var), np.dot(prec_chol, mean_0), axis=0)
    # gaussian posterior ...
    result["mean"] = la.lstsq(Xt, tt)[0]
    _, r = la.qr(Xt, mode='economic')
    result["sqrt_prec"] = r
    result["prec"] = np.dot(r.T, r)
    result["var"] = var
    return result 


def predictive(samples, var, mean, sqrt_prec, **params):
    """
    """
    result = {}
    # sqrt_eqk -- square root of equivalent kernel, 
    # compare to Bishop, p. 159.
    # sqrt_prec is _upper-triangular_
    # choleksy factor of precision
    sqrt_eqk = la.solve_triangular(sqrt_prec, samples.T, trans=1).T
    result["mean"] = np.dot(samples, mean)
    result["var"] = var + np.sum(sqrt_eqk**2, axis=1)
    return result 


def bayesian_online(x, y, var, mean, cov):
    """
    Standard recursive least squares.

    Note: Numerically problematic.
    Missing QR/Cholesky Up/Downdate code.
    """
    tmp = np.dot(cov, x.T)
    scalar = 1./var + np.dot(x, tmp)
    # Update covariance
    cov -= np.dot(tmp, tmp.T)/scalar
    # Update mean, without using Kalman Gain
    mean += var * (y - np.dot(x, mean)) * np.dot(cov, x.T)
    return {"mean": mean, "cov": cov}


def robust(X, t, mean_0, prec_0, 
        a, b, var, eps=10**-4, **params):
    """
    """
    prec_chol = la.cholesky(prec_0)
    Xt = np.append(X, prec_chol, axis=0)
    tt = np.append(t, np.dot(prec_chol, mean_0), axis=0)
    n, d = Xt.shape
    w = var*np.ones((n,1))
    w[:-d] = a/b
    # a only serves as shifted constant
    a += 0.5
    result = {}
    ll_old = -np.inf
    lls = []
    while True:
        # estimate gaussian for mean
        q, r = la.qr(np.sqrt(w/var)*Xt, mode='economic')
        del q
        mean = la.lstsq(np.sqrt(w/var)*Xt, np.sqrt(w/var)*tt)[0]
        # estimate posterior weights 
        # se -- squared error between prediction and target
        se = (t - np.dot(X, mean))**2
        # sqrt_eqk -- square root of equivalent kernel
        sqrt_eqk = la.solve_triangular(r, X.T, trans=1).T
        mhlb = np.sum(sqrt_eqk**2, axis=1)[:, np.newaxis]
        # a stays constant, according to formula in paper
        b_new = b + (se + mhlb)/(2*var)
        w_new = a/b_new
        # M-Step, estimate sigma
        var = np.mean(w_new * (se + mhlb))
        w[:-d] = w_new
        w[-d:] = var
        # maximize loglikelihood -> track its progress
        # leaving out constant terms
        ll = np.log(var)/2
        ll -= np.sum(w[:-d] * (se + mhlb))/(2*var)
        # entropies
        # gaussian of mean, no constant terms
        ll += (d-n) * np.sum(np.diag(r))
        # gamma dist, again no constants
        ll += 2 * np.sum(np.log(b_new))
        if (ll < ll_old):
            print "Loglikelihood decreased to %f!" % ll
            print "This can happen after the first iteration!"
            lls.append(ll)
            ll_old = ll
        elif ((ll - ll_old) < eps):
            result["mean"] = mean
            result["sqrt_prec"] = r
            result["weights"] = w[:-d]
            result["var"] = var
            result["logl"] = lls
            break
        else:
            print "LogL: ", ll
            lls.append(ll)
            ll_old = ll
    a -= 0.5
    return result


def robust_online(x, t, lmbd, mean_0, prec_0,
        a, b, var, eps=10**-4, **params):
    """
    """
    tmp = np.dot(cov, x.T)
    sqr = np.dot(x, tmp)
    diff = t - np.dot(x.T, mean)
    while True:
        scalar = lmbd/w + sqr
        _cov = _cov - np.dot(tmp, tmp.T)/scalar
        _mean = mean + diff * w * tmp
        se = (t - np.dot(x.T, _mean))**2
        mhlb = np.dot(x, np.dot(_cov, x))
        # a stays constant, according to formula in paper
        b_new = b + (se + mhlb)/(2*var)
        w_new = a/b_new
        # sigma


def demo_bayesian(nos, nob, var=0.1, width=0.4):
    samples = np.random.rand(nos, 1)
    centers = np.linspace(0, 1, nob)
    width = 4./10
    X = np.zeros((nos, nob+1))
    for i, s in enumerate(samples):
        X[i, 0] = 1
        for j, c in enumerate(centers):
            X[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)
    y = np.sin(2*np.pi*samples) + np.sqrt(var)*np.random.randn(nos, 1)
    #
    m_0 = np.zeros((nob+1, 1))
    p_0 = 0.0001 * np.eye(nob+1)
    bys = bayesian(X, y, 0.1, m_0, p_0)
    # online
    cov = 1000. * np.eye(nob + 1)
    mean = np.zeros((nob + 1, 1))
    online = {"mean": mean, "cov": cov}
    for i, r in enumerate(X):
        online = bayesian_online(np.atleast_2d(r), y[i], 0.1, **online)
    #
    tests = np.linspace(0, 1, 100)
    Xt = np.zeros((tests.shape[0], nob+1))
    for i, s in enumerate(tests):
        Xt[i, 0] = 1
        for j, c in enumerate(centers):
            Xt[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)
    pred = predictive(Xt, **bys)
    online["sqrt_prec"] = la.cholesky(la.inv(online["cov"]))
    online["var"] = var
    onpred = predictive(Xt, **online)
    import pylab
    pylab.plot(samples, y, 'ro')
    pylab.plot(tests, pred["mean"], 'g')
    pylab.fill_between(tests, pred["mean"].ravel() + np.sqrt(pred["var"]), 
            pred["mean"].ravel() - np.sqrt(pred["var"]), alpha=0.1, color='g')
    pylab.plot(tests, onpred["mean"], 'y')
    pylab.fill_between(tests, onpred["mean"].ravel() + np.sqrt(onpred["var"]), 
            onpred["mean"].ravel() - np.sqrt(onpred["var"]), alpha=0.1, color='y')
    pylab.plot(tests, np.sin(2*np.pi*tests), 'r', lw=2)


def demo_robust(nos, noo, nob, var=1., k=3, width=0.4):
    """
    Use _nos_ many normal samples and
    _noo_ many outlier samples on the
    sin function. Use _nob_ many
    RBF basis functions of _width_.
    """
    import random, pylab
    # Generate 1d inputs
    samples = np.random.rand(nos+noo, 1)

    # Make design matrix -> RBF's are needed
    # RBF's are centered equi-distant
    centers = np.linspace(0, 1, nob)
    # X is the design matrix
    X = np.zeros((nos+noo, nob+1))
    for i, s in enumerate(samples):
        X[i, 0] = 1
        for j, c in enumerate(centers):
            X[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)

    # Generate targets
    y = np.sin(2*np.pi*samples) + np.random.randn(nos+noo, 1)
    outliers = random.sample(xrange(nos+noo), noo)
    y[outliers] = np.sin(2*np.pi*samples[outliers]) + (k*k*np.random.rand(noo, 1))*np.sqrt(var)
    # normalize
    ym = np.mean(y)
    ystd = np.std(y)
    ynormed = (y-ym)/ystd

    # Priors for EM algorithm
    a = np.ones((nos+noo, 1))
    b = np.ones((nos+noo, 1))
    m_0 = np.zeros((nob+1, 1))
    p_0 = 0.001 * np.eye(nob+1)

    # robust bayesian lr
    brlr = robust(X, ynormed, m_0, p_0, a, b, var)

    # 'simple' bayesian lr
    bys = bayesian(X, ynormed, var, m_0, p_0)

    # Test set is [0,1]
    tests = np.linspace(0, 1, 100)
    # Design matrix for testset
    Xt = np.zeros((tests.shape[0], nob+1))
    for i, s in enumerate(tests):
        Xt[i, 0] = 1
        for j, c in enumerate(centers):
            Xt[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)

    # prediction for robust model
    r_pred = np.dot(Xt, brlr["mean"])
    
    # prediction for baysian model
    b_pred = predictive(Xt, **bys)

    # Plotting business -- boring
    pylab.plot(samples, y, 'ro')
    # Show outliers in black
    pylab.plot(samples[outliers], y[outliers], 'ko')
    pylab.plot(tests, np.sin(2*np.pi*tests), 'r', lw=2)
    pylab.plot(tests, ystd*r_pred + ym, 'g', lw=2)
    pylab.plot(tests, ystd*b_pred["mean"] + ym, 'b', lw=2)
