"""

"""


import numpy as np


from misc import logsumexp, sigmoid


def xe(z, targets, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return np.argmax(z, axis=1)
    _xe = z - logsumexp(z, axis=1)
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
        return weight*np.mean(err**2)+addon, 2*weight*err/(n*m)
    else:
        # only return reconstruction error 
        return weight*np.mean(err**2)+addon


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


def zero_one(z, targets):
    """
    """
    return np.sum(z!=targets)
