"""Mixture of Gaussians.

(Also runs under 'Gaussian Mixture Model, GMM.')

Also tucked in: kmeans, minibatch kmeans.
"""


import numpy as np
import scipy.linalg as la
from scipy.sparse import csc

from misc import norm_logprob, logsumexp
from metric import euc_dist


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
        assert c is not None, "M is None, need number of classes c."
        assert D is not None, "M is None, need observation dimension D."
        M = 5*np.random.randn(c, D)
    
    if Cov is None:
        assert c is not None, "Cov is None, need number of classes c."
        assert D is not None, "Cov is None, need observation dimension D"
        # This is not yet a covariance matrix!
        Cov = np.random.randn(c, D, D)

    if pi is None:
        assert c is not None, "pi is None, need number of classes c."
        pi = np.random.rand(c)
        pi = pi/sum(pi)

    c, D, _ = Cov.shape
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
        Cov[j] = (Cov[j] + Cov[j].T)/5 + np.eye(D)
        chol = la.cholesky(Cov[j])
        samples[j_idx, :] = np.dot(tmp, chol) + M_c
    return samples, classes, M, Cov, pi


def kmeans(X, K, maxiters, M=None, eps=1e-3):
    """Standard k-means.

    _X_ is data rowwise. _K_ is the number of
    clusters. _M_ is the set of centers.

    Implementation tries to save some computation cycles.
    """
    N, d = X.shape
    if M is None:
        tmp = np.random.permutation(N)
        M = X[tmp[:K]].copy()

    costs = []
    last = np.inf
    X_sq_sum = np.sum(X**2)
    for i in xrange(maxiters):
        # see metric.py, but here: don't need squares from
        # X, because _minimal_ cost over K is independent from it.
        cost = -2*np.dot(X, M.T) + np.sum(M**2, axis=1)
        idx = np.argmin(cost, axis=1)
        cost = cost[xrange(N), idx]
        costs.append(X_sq_sum + np.sum(cost))

        if (last - costs[-1]) < eps:
            break
        last = costs[-1]
        # Determine new centers
        # Sparseification from Jakob Verbeek's kmeans code,
        # http://lear.inrialpes.fr/~verbeek/software.php
        ind = csc.csc_matrix( (np.ones(N), (idx, xrange(N))), shape=(K, N))
        M = ind.dot(X)
        weights = np.array(ind.sum(axis=1))
        # Handle problem: no points assigned to a cluster
        zeros_idx = (weights.ravel()==0)
        zeros = np.sum(zeros_idx)
        tmp = np.random.permutation(N)
        M[zeros_idx, :] = X[tmp[:zeros]].copy()
        weights[zeros_idx] = 1
        M /= weights
    return M, costs


def minibatch_k_means(X, k, mbsize, iters):
    """
    Minibatch kmeans according to
    Sculley's paper.

    Parameters
    ----------
    X: array, shape (n_samples, n_features)
        The data matrix.

    k: int
        Number of centers.

    mbsize: int
        Size of minibatch.

    iters: int
        Number of iterations.

    Returns
    -------
    centers: array, shape (k, n_features)
        Computed centers.

    inertia: float
        The value of the inertia criterion 
        given _centers_.
    """

    samples = X.shape[0]
    # Initialize centers
    seed = np.argsort(np.random.rand(samples))[:k]
    centers = X[seed]
    v = np.ones(k)
    
    for i in xrange(iters):
        sample = np.argsort(np.random.rand(samples))[:mbsize]
        M = X[sample]
        d = np.argmin(euc_dist(M, centers, squared=True), axis=1)
        for j in xrange(mbsize):
            c = d[j]
            v[c] += 1
            eta = 1./v[c]
            centers[c] = (1-eta) * centers[c] + eta * M[j]
    
    distances = euc_dist(X, centers, squared=True)
    closest_center = np.argmin(distances, axis=1)
    inertia_ = distances[xrange(samples), closest_center].sum()
    
    return centers, inertia_


def skmeans(X, k, epochs, M=None, btsz=100, lr=0.001):
    """
    Synchronous k-Means. From
    "Learning to encode motion using spatio-temporal synchrony",
    by Konda, Memisevic and Michalski.
    """
    n, d = X.shape
    if M is None:
        M = np.random.standard_normal((d, k))
        Mlength = np.sqrt(np.sum(M**2, axis=0) + 1e-6)
        M /= Mlength
    mx = np.zeros((btsz, k))
    mx_sq = np.zeros((btsz, k))
    Mlength = np.zeros(k)
    for ep in xrange(epochs):
        _cost = 0
        for i in xrange(0, n, btsz):
            mx[...] = 0.
            mx_sq[...] = 0.
            _x = X[i:i+btsz]
            _n, __n = _x.shape
            sprod = np.dot(_x, M)
            cost = sprod**2
            idx = np.argmax(cost, axis=1)
            mx[xrange(_n), idx] = sprod[xrange(_n), idx]
            mx_sq[xrange(_n), idx] = cost[xrange(_n), idx]
            mx_sq_sum = mx_sq.sum(axis=0)
            M += lr*(np.dot(_x.T, mx) - mx_sq_sum*M)
            Mlength[:] = np.sqrt(np.sum(M**2, axis=0) + 1e-6)
            M /= Mlength
            _cost += np.sum((_x - np.dot(mx, M.T))**2)
        print "Epoch: ", ep, "; Cost: ", _cost
    return M, cost, 0

def kmeans_np(X, lmbda, M=None):
    """Non-parametric kmeans.

    _X_ is input data, rowwise. _lmbda_ controls
    tradeoff between standard kmeans and cluster
    penalty term.

    See http://www.cs.berkeley.edu/~jordan/papers/kulis-jordan-icml12.pdf
    """
    N, d = X.shape
    if M is None:
        M = np.mean(X, axis=0).reshape(1, d)
    k = M.shape[0] - 1 
    X_sq_sum = np.sum(X**2, axis=1)
    ind = np.zeros(N)
    old_ind = ind.copy()
    tmp = 0
    iters = 1
    while True:
        print "Iteration ", iters
        iters = iters + 1
        for i in xrange(N):
            tmp = -2*np.dot(X[i], M.T) + np.sum(M**2, axis=1)
            idx = np.argmin(tmp)
            if (X_sq_sum[i] + tmp[idx]) > lmbda:
                k = k + 1
                M = np.append(M, X[i].copy().reshape(1, d), axis=0)
                ind[i] = k
                print "Adding cluster for ", i, k
            else:
                ind[i] = idx
        if np.all(old_ind == ind):
            break
        # see kmeans above
        ind_all = csc.csc_matrix((np.ones(N), (ind, xrange(N))), shape=(k+1, N))
        M = ind_all.dot(X)
        M /= np.array(ind_all.sum(axis=1))
        old_ind = ind
        ind = np.zeros(N)
    return M, np.array(ind_all.sum(axis=1)).ravel()
