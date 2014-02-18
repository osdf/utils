"""

"""


import numpy as np


from misc import logsumexp, sigmoid


def xe(z, targets, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return np.argmax(z, axis=1)
    _xe = z - np.atleast_2d(logsumexp(z, axis=1)).T
    n, _ = _xe.shape
    xe = -np.mean(_xe[np.arange(n), targets])
    if error:
        err = np.exp(_xe)
        err[np.arange(n), targets] -= 1
        #score + error
        return xe+addon, err/n
    else:
        return xe+addon


def ssd(z, targets, weight=0.5, predict=False, error=False, addon=0):
    """
    Sum-of-squares difference (ssd).
    """
    if predict:
        return z
    n, m = z.shape
    err = z - targets
    if error:
        # rec. error + first deriv
        return weight*np.mean(np.sum(err**2, axis=1))+addon, 2*weight*err/n
    else:
        # only return reconstruction error 
        return weight*np.mean(np.sum(err**2, axis=1))+addon


def mia(z, targets, predict=False, error=False, addon=0):
    """
    Multiple independent attributes.

    Feed model output _z_ through logistic to get
    bernoulli distributed variables. 
    """
    bern = sigmoid(z)
    if predict:
        return bern
    n, _ = bern.shape
    # loss is binary cross entropy
    # for every output variable
    bce = -(targets*np.log(bern) + (1-targets)*np.log(1-bern))
    bce = np.mean(np.sum(bce, axis=1))
    if error:
        return bce+addon, (bern-targets)/n
    else:
        return bce+addon
    

def bKL(x, y):
    """
    Kullback-Leibler divergence between two
    bernoulli random variables _x_ and _y_.
    """
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))


def drLim(z, targets, p_margin, n_margin=10, predict=False, error=False, addon=0):
    """
    Dimensionality Reduction by Learning an Invariant Metric.
    """
    pairs, d = z.shape
    z = z.reshape(pairs//2, 2*d)
    dist = z[:, :d] - z[:, d:]
    dist *= dist
    dist = np.sqrt(dist.sum(axis=1), + eps)
    dist = np.repeat(dist, 2, axis=0)
    z = z.reshape(pairs, d)
    z[1::2] *= -1
    n_margin_active = 1.0*(dist < n_margin)
    err = targets*z + (1-targets)*n_margin_active*z
    if error:
        return 0 ,err
    else:
        return 0


def zero_one(z, targets):
    """
    """
    return np.sum(z!=targets)
