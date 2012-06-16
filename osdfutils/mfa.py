"""Mixture of Factor Analysers.

This is based on 'The EM algorithm for Mixtures of Factor Analyzers'
by Ghahramani, Hinton, see http://www.cs.toronto.edu/~hinton/absps/tr-96-1.pdf.

Furthermore, I used Jakob Verbeeks matlab implementation to check and
finetune (== fix mistakes and clean up messy code) my original implementation.
See http://lear.inrialpes.fr/~verbeek/software.php
"""


import numpy as np
import scipy.linalg as la


from misc import norm_logprob, logsumexp


SMALL = 1e-8


def mfa(X, hdim, C, maxiters, W=None, M=None, psi=None, pi=None, eps=1e-2):
    """Fit a Mixture of FA.

    _X_ is dataset in _rows_. _hdim_ is the
    latent dimension, the same for all _C_
    classes.
    """
    # pre calculation of some 'constants'.
    N, d = X.shape
    Ih = np.eye(hdim)
    ll_const = -d/2. * np.log(2*np.pi)
    X_sq = X**2

    if W is None:
        W = np.random.randn(C, hdim, d)
    if M is None:
        tmp = np.random.permutation(N)
        M = X[tmp[:C]].copy()
    if psi is None:
        psi = 100*np.var(X)*np.ones((C, d))
    if pi is None:
        pi = np.ones(C)/C


    # pre allocating some helper memory
    E_z = np.zeros((C, N, hdim))
    Cov_z = np.zeros((C, hdim, hdim))
    # store loglikelihood
    ll = np.zeros((C, N))

    last_ll = -np.inf
    loglike = []
    for i in xrange(maxiters):
        for c in xrange(C):
            # W_c is hdim x d 
            W_c = W[c]
            mu_c = M[c]
            # psi_c is D
            psi_c = psi[c]
            fac = W_c/psi_c
            # see Bishop, p. 93, eq. 2.117 
            cov_z = la.inv(Ih + np.dot(fac, W_c.T))
            tmp = np.dot(X - mu_c, fac.T)
            # latent expectations
            E_z[c, :, :] = np.dot(tmp, cov_z)
            # latent _covariance_
            Cov_z[c, :, :] = cov_z
            # loglikelihood
            # woodbury identity
            inv_cov_x = np.diag(1./psi_c) - np.dot(fac.T, np.dot(cov_z, fac))
            _, _det = np.linalg.slogdet(inv_cov_x)
            tmp = np.dot(X-mu_c, inv_cov_x)
            # integrating out latent z's -> again, Bishop, p. 93, eq. 2.115 
            ll[c, :] = np.log(pi[c]) + ll_const + 0.5 * _det - 0.5 * np.sum(tmp*(X-mu_c), axis=1)
        # posterior class distribution given data
        posteriors = norm_logprob(ll, axis=0)
        # loglikelihood over all datapoints
        ll_sum = np.sum(logsumexp(ll, axis=0))
        loglike.append(ll_sum)

        if ll_sum - last_ll < eps:
            break
        last_ll = ll_sum

        for c in xrange(C):
            z = np.append(E_z[c, :, :], np.ones((N, 1)), axis=1)
            wz = posteriors[c][:, np.newaxis] * z
            wzX = np.dot(wz.T, X)
            wzz = np.dot(wz.T, z)
            N_c = posteriors[c].sum()
            wzz[:hdim, :hdim] += N_c * Cov_z[c, :, :]

            sol = la.lstsq(wzz, wzX)[0]

            M[c, :] = sol[hdim, :]
            W[c, :, :] = sol[:hdim, :]
            psi[c, :] = (np.dot(posteriors[c], X_sq) - np.sum(sol*wzX, axis=0))/N_c
            psi[c, :] = np.maximum(psi[c, :], SMALL)
            pi[c] = N_c/N
    return W, M, psi, pi, loglike


def sampling(n, W=None, M=None, psi=None, pi=None, c=None, D=None, h=None):
    """Produce _n_ samples from an MFA.

    Usually the caller passes in _W_ (weight matrices for every component,
    dimension c x D x h), _M_ (means, c x D), _psi_ (noise variance,
    c x D) and _pi_ (class probabilities, c x 1). Hereby is _c_ the
    number of mixture components, _D_ is the dimension of the observed
    variables and _h_ is the latent dimension. 
    
    If one of these parameters is not given, _c_, _D_ and _h_ must be provided.

    Because of this case, all parameters are returned, too.
    """
    if W is None:
        assert c is not None, "W is None, need number of classes."
        assert D is not None, "W is None, need observation dimension D"
        assert h is not None, "W is None, need latent dimension h."
        W = np.random.randn(c, D, h)
    
    if M is None:
        assert c is not None, "M is None, need number of classes."
        assert D is not None, "M is None, need observation dimension D."
        M = 10*np.random.randn(c, D)

    if psi is None:
        assert c is not None, "psi is None, need number of classes."
        assert D is not None, "psi is None, need observation dimension D."
        psi = 0.1 * np.random.randn(c, D)

    if pi is None:
        assert c is not None, "pi is None, need number of classes."
        pi = np.random.rand(pi)
        pi = pi/sum(pi)

    # shape of W for every component: hidden x visible,
    # data is rowwise.
    c, h, D = W.shape
    samples = np.zeros((n, D))

    # sample classes
    classes = np.random.rand(n)
    csum = np.cumsum(pi)
    classes = np.sum(classes[:, np.newaxis] > csum, 1)

    for j in xrange(c):
        j_idx = classes==j
        nc = np.sum(j_idx)
        W_c = W[j]
        M_c = M[j]
        psi_c = psi[c]
        latent = np.random.randn(nc, h)    
        noise = np.sqrt(psi_c) * np.random.randn(nc, D)
        samples[j_idx, :] = np.dot(latent, W_c) + M_c + noise
    return samples, latent, classes, W, M, psi, pi


def test():
    """
    """
    W = np.zeros((2, 2, 3))
    W[0, :, :] = np.array([[0.5377, 0.8622],[1.8339, 0.3188],[-2.2588, -1.3077]]).T
    W[1, :, :] = np.array([[-0.4336, 2.7694],[0.3426, -1.3499],[3.5784, 3.0349]]).T

    M = np.array([[1, -1, 0.5], [-2, 0, 1]])
    psi = np.array([[0.25, 1./9, 1./4], [1./16, 1, 1./9]])
    latent = np.array([0.2761 ,-0.2612 ,0.4434 ,0.3919 ,-1.2507 ,-0.9480 ,-0.7411 ,-0.5078 ,-0.3206 ,0.0125 ,-3.0292 ,-0.4570 ,1.2424 ,-1.0667 ,0.9337 ,0.3503 ,-0.0290 ,0.1825 ,-1.5651 ,-0.0845 ,1.6039 ,0.0983 ,0.0414 ,-0.7342 ,-0.0308 ,0.2323 ,0.4264 ,-0.3728 ,-0.2365 ,2.0237 ,-2.2584 ,2.2294 ,0.3376 ,1.0001 ,-1.6642 ,-0.5900 ,-0.2781 ,0.4227 ,-1.6702 ,0.4716 ,-1.2128 ,0.0662 ,0.6524 ,0.3271 ,1.0826 ,1.0061 ,-0.6509 ,0.2571 ,-0.9444 ,-1.3218 ,0.9248 ,0.0000 ,-0.0549 ,0.9111 ,0.5946 ,0.3502 ,1.2503 ,0.9298 ,0.2398 ,-0.6904 ,-0.6516 ,1.1921 ,-1.6118 ,-0.0245 ,-1.9488 ,1.0205 ,0.8617 ,0.0012 ,-0.0708 ,-2.4863 ,0.5812 ,-2.1924 ,-2.3193 ,0.0799 ,-0.9485 ,0.4115 ,0.6770 ,0.8577 ,-0.6912 ,0.4494 ,0.1006 ,0.8261 ,0.5362 ,0.8979 ,-0.1319 ,-0.1472 ,1.0078 ,-2.1237 ,-0.5046 ,-1.2706 ,-0.3826 ,0.6487 ,0.8257 ,-1.0149 ,-0.4711 ,0.1370 ,-0.2919 ,0.3018 ,0.3999 ,-0.9300])
    latent = latent.reshape(50, 2)
    noise = np.array([-0.0436 ,0.5824 ,-1.0065 ,0.0645 ,0.6003 ,-1.3615 ,0.3476 ,-0.1818 ,-0.9395 ,-0.0375 ,-1.8963 ,-2.1280 ,-1.1769 ,-0.9905 ,-1.1730 ,-1.7254 ,0.2882 ,-1.5942 ,0.1102 ,0.7871 ,-0.0022 ,0.0931 ,-0.3782 ,-1.4827 ,-0.0438 ,0.9608 ,1.7382 ,-0.4302 ,-1.6273 ,0.1663 ,0.3763 ,-0.2270 ,-1.1489 ,2.0243 ,-2.3595 ,-0.5100 ,-1.3216 ,-0.6361 ,0.3179 ,0.1380 ,-0.7107 ,0.7770 ,0.6224 ,0.6474 ,-0.4256 ,1.0486 ,0.6607 ,2.5088 ,1.0635 ,1.1569 ,0.0530 ,-1.2884 ,-0.3712 ,-0.7578 ,-0.5640 ,0.5551 ,-0.5568 ,-0.8951 ,-0.4093 ,-0.1609 ,0.4093 ,-0.9526 ,0.3173 ,0.0780 ,1.3244 ,-0.2132 ,-0.1345 ,-1.1714 ,-1.3853 ,0.3105 ,-0.2495 ,0.5037 ,-0.8927 ,1.9085 ,0.1222 ,1.0470 ,-0.2269 ,-0.1625 ,0.6901 ,0.5558 ,-1.1203 ,-1.5327 ,-1.0979 ,-1.4158 ,0.0596 ,-0.4113 ,-0.3680 ,-1.3610 ,0.7796 ,0.4394 ,-0.0896 ,1.0212 ,-0.8740 ,0.4147 ,0.3484 ,0.3493 ,-0.7292 ,0.3268 ,-0.5149 ,-0.8964 ,-1.2033 ,1.0378 ,-0.8459 ,-0.1729 ,-1.2087 ,-0.2971 ,-3.2320 ,-1.0870 ,-1.4264 ,-1.0145 ,-0.2133 ,-0.3253 ,1.9444 ,-0.5718 ,-0.2500 ,-1.5693 ,-0.4774 ,-1.3380 ,0.0303 ,0.8531 ,0.4043 ,-0.7006 ,-1.6305 ,1.4600 ,2.0500 ,0.1205 ,-0.9899 ,1.1978 ,-0.5927 ,-0.4698 ,0.8864 ,-1.3852 ,-1.9568 ,0.4207 ,0.4007 ,0.0951 ,0.4967 ,1.0822 ,0.9704 ,-0.5686 ,0.8100 ,0.1732 ,-0.5055 ,-1.1933 ,0.6470 ,-0.3536 ,0.0464 ,-0.7929 ,-1.5505 ,0.1716])
    noise = noise.reshape(50, 3)
    classes = np.array([1, 1 ,0 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ,1 ,0 ,1 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,0 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,0])

    D = 3
    N = 50
    samples = np.zeros((50, 3))
    for j in xrange(2):
        j_idx = classes==j
        _latent = latent[j_idx, :]
        W_c = W[j]
        M_c = M[j]
        psi_c = psi[j]
        samples[j_idx, :] = np.dot(_latent, W_c) + M_c + np.sqrt(psi_c) * noise[j_idx, :]
    return samples
