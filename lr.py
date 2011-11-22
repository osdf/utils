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


def bayesian(X, t, sigma, mean_0, prec_0, **params):
    """

    """
    result = {}
    prec_chol = la.cholesky(prec_0)
    # build up augmented designmatrix/targetvector
    Xt = np.append(X/sigma, prec_chol, axis=0)
    tt = np.append(t/sigma, np.dot(prec_chol, mean_0), axis=0)
    # gaussian posterior ...
    result["mean"] = la.lstsq(Xt, tt)[0]
    _, r = la.qr(Xt, mode='economic')
    result["sqrt_prec"] = r
    result["prec"] = np.dot(r.T, r)
    return result 


def predictive(samples, sigma, mean, sqrt_prec, **params):
    """
    """
    # sqrt_eqk -- square root of equivalent kernel, 
    # compare to Bishop, p. 159
    sqrt_eqk = la.solve_triangular(sqrt_prec, samples.T, trans=1).T
    result = {}
    result["mean"] = np.dot(samples, mean)
    result["var"] = sigma + np.sum(sqrt_eqk**2, axis=1)
    return result 


def robust(X, t, mean_0, prec_0, a, b, var=1., **params):
    """
    """
    prec_chol = la.cholesky(prec_0)
    Xt = np.append(X, prec_chol, axis=0)
    tt = np.append(t, np.dot(prec_chol, mean_0), axis=0)
    n, d = Xt.shape
    w = np.ones(n,1)
    w[:-d] = a/(b*var)
    # a only serves as shifted constant
    a += 0.5
    result = {}
    ll_old = -np.inf
    while True:
        # estimate gaussian for mean
        mean = la.lstsq(np.sqrt(w)*Xt, np.sqrt(w)*tt)[0]
        q, r = la.qr(np.sqrt(w/var)*Xt, mode='economic')
        del q
        # estimate gamma 
        # se -- squared error between prediction and target
        se = (t - np.dot(X, mean))**2
        # sqrt_eqk -- square root of equivalent kernel
        sqrt_eqk = la.solve_triangular(r, X.T, trans=1).T
        mhlb = np.sum(sqrt_eqk**2, axis=1)
        # check here: officially, first compute w_new
        var = np.mean(se) + np.mean(mhlb)
        # a stays constant, according to formula in paper
        b_new = b + (se + mhlb)/(2*var)
        w_new = a/b_new
        w[:-d] = w_new/var
        # maximize loglikelihood -> track its progress
        # leaving out constant terms
        ll = -n * np.ln(var)/2
        ll -= 0.5 * np.sum(w[:-d] * se)
        # entropies
        # gaussian of mean, no constant terms
        ll += np.sum(np.diag(r))
        ll -= 2 * np.sum(np.ln(b_new))
        if ((ll - ll_old) < 0.1):
            result["mean"] = mean
            result["sqrt_prec"] = r
            result["weights"] = w[:-d]
            result["var"] = var
            break
        if (ll < ll_old):
            print "STRESS"
        else:
            ll_old = ll
    a -= 0.5
    return result


def demo_bayesian(nos, nob):
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


def demo_robust(nos, nob):
    samples = np.random.rand(nos, 1)
    centers = np.linspace(0, 1, nob)
    width = 4./10
    X = np.zeros((nos, nob+1))
    y = np.zeros((nos, 1))
    var = 1
    for i, s in enumerate(samples):
        X[i, 0] = 1
        for j, c in enumerate(centers):
            X[i, 1+j] = np.exp(-0.5 * ((s-c)/width)**2)
            w = np.random.gamma(1,1)
            y[i] = np.sin(2*np.pi*s) + np.sqrt(var/w)*np.random.randn(1)
    tests = np.linspace(0, 1, 100)
    import pylab
    pylab.plot(samples, y, 'ro')
    pylab.plot(tests, np.sin(2*np.pi*tests), 'r', lw=2)
