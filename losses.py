"""

"""


import numpy as np


from misc import logsumexp, sigmoid


def xe(z, targets, predict=False, error=False, addon=0):
    """
    """
    if predict:
        return np.argmax(z, axis=1)
    #
    _xe = z - logsumexp(z, axis=1)
    n, _ = _xe.shape
    xe = -np.sum(_xe[np.arange(n), targets])
    if error:
        err = np.exp(_xe)
        err[np.arange(n), targets] -= 1
        #score + error
        return xe+addon, err
    else:
        return xe+addon


def ssd(z, targets, weight=0.5, predict=False, error=False, addon=0):
    """
    Sum-of-squares difference (ssd).
    """
    if predict:
        return z
    #
    err = z - targets
    if error:
        # rec. error + first deriv
        return weight*np.sum(err**2)+addon, 2*weight*err
    else:
        # only return reconstruction error 
        return weight*np.sum(err**2)+addon


def mia(z, targets, predict=False, error=False, addon=0):
    """
    Multiple independent attributes.

    Feed model output _z_ through logistic to get
    bernoulli distributed variables. 
    """
    bern = sigmoid(z)
    if predict:
        return bern
    # loss is binary cross entropy
    # for every output variable
    bce =  -( targets*bern.log() + (1-targets)*(1-bern).log() ).sum()
    if error:
        return bce+addon, z-targets
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
