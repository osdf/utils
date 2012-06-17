"""Mixture of Gaussians.

(Also runs under 'Gaussian Mixture Model, GMM.')
"""


import numpy as np
import scipy.linalg as la


from misc import norm_logprob, logsumexp


LOGSMALL = -4


def mog(X, C, maxiters, M=None, Cov=None, pi=None, eps=1e-2):
    """Fit a MoG.

    _X_ is dataset in _rows_. _C_ is the number
    of clusters.
    """
    N, d = X.shape
    ll_const = -d/2. * np.log(2*np.pi)

    if M is None:
        tmp = np.random.permutation(N)
        M = X[tmp[:C]].copy()
    if Cov is None:
        diag = np.mean(np.diag(np.cov(X, rowvar=0)))*(np.abs(np.random.randn())+1)
        Cov = (diag * np.eye(d)).reshape(1, d, d).repeat(C, axis=0)
    if pi is None:
        pi = np.ones(C)/C

    ll = np.zeros((C, N))
    last_ll = -np.inf
    loglike = []
    
    for i in xrange(maxiters):
        for c in xrange(C):
            cov_c = Cov[c]
            chol = la.cholesky(cov_c)
            # not exactly log det, factor 2 missing
            # but ok here, because no 0.5 factor below
            logdet = np.sum(np.log(np.diag(chol)))

            mu_c = M[c]

            # mahalanobis distance 
            mhlb = (la.solve_triangular(chol, (X - mu_c).T, trans=1).T**2).sum(axis=1)
            # note missing 0.5 before logdet
            ll[c, :] = np.log(pi[c]) + ll_const - logdet - 0.5 * mhlb
        # posterior class distribution given data
        posteriors = norm_logprob(ll, axis=0)
        # loglikelihood over all datapoints
        ll_sum = np.sum(logsumexp(ll, axis=0))
        loglike.append(ll_sum)

        if ll_sum - last_ll < eps:
            break
        last_ll = ll_sum

        for c in xrange(C):
            N_c = posteriors[c].sum()
            M[c, :] = np.sum(posteriors[c][:, np.newaxis] * X, axis=0)/N_c
            tmp = X - M[c]
            _cov = np.dot(posteriors[c]*tmp.T, tmp)/N_c
            # check if on trip to singularity
            # probably could be done a bit more clever,
            # in combination with E-step above (cholesky already
            # here and cache)
            _, _det = np.linalg.slogdet(_cov)
            if _det > LOGSMALL:
                Cov[c, :, :] = _cov
            pi[c] = N_c/N
    return M, Cov, pi, loglike


def sampling(n, M=None, Cov=None, pi=None, c=None, D=None):
    """Produce _n_ samples from an MoG.

    Usually the caller passes in _M_ (means, c x D), and _pi_ 
    (class probabilities, c x 1). Hereby is _c_ the
    number of mixture components, _D_ is the dimension of the observed
    variables. 
    
    If one of these parameters is not given, _c_, and _D_.

    Because of this case, all parameters are returned, too.
    """
    if M is None:
        assert c is not None, "M is None, need number of classes."
        assert D is not None, "M is None, need observation dimension D."
        M = 5*np.random.randn(c, D)
    
    if Cov is None:
        assert c is not None, "Cov is None, need number of classes."
        assert D is not None, "Cov is None, need observation dimension D"
        # This is not yet a covariance matrix!
        Cov = np.random.randn(c, D, D)

    if pi is None:
        assert c is not None, "pi is None, need number of classes."
        pi = np.random.rand(c)
        pi = pi/sum(pi)

    c, D, D = Cov.shape
    # data is rowwise.
    samples = np.zeros((n, D))

    # sample classes
    classes = np.random.rand(n)
    csum = np.cumsum(pi)
    classes = np.sum(classes[:, np.newaxis] > csum, 1)

    for j in xrange(c):
        j_idx = classes==j
        nc = np.sum(j_idx)
        tmp = np.random.randn(nc, D)
        M_c = M[j]
        # Covariance construction
        Cov[j] = (Cov[j] + Cov[j].T + 10*np.eye(D))/10
        chol = la.cholesky(Cov[j])
        samples[j_idx, :] = np.dot(tmp, chol) + M_c
    return samples, classes, M, Cov, pi


def test(n=100):
    """
    """
    o = np.random.randn()
    var = 2
    samples = np.random.randn(n, 2)
    dists = o + var*np.random.randn(n)
    thetas = np.random.randn(n) * 2 * np.pi
    samples[:, 0] = dists * np.cos(thetas)
    samples[:, 1] = dists * np.sin(thetas)
    return samples
