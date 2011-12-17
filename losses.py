"""

"""


import numpy as np


from misc import logsumexp


def xe(z, targets, predict=False, error=False):
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
        return xe, err
    else:
        return xe


def ssd(z, targets, weight=0.5, predict=False, error=False):
    """
    """
    if predict:
        return z
    #
    err = z - targets
    if error:
        # rec. error + first deriv
        return weight*np.sum(err**2), 2*weight*err
    else:
        # only return reconstruction error 
        return weight*np.sum(err**2)


def mia():
    """
    Multiple independent attributes.
    """
    pass


def zero_one(z, targets):
    """
    """
    return np.sum(z!=targets)
