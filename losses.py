"""

"""


import numpy as np


from misc import logsumexp


def score_xe(z, targets, predict=False, error=False):
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


def score_ssd(z, targets, predict=False, error=False):
    """
    """
    if predict:
        return z
    #
    err = z - targets
    if error:
        # score + error
        return 0.5*np.sum(err**2), err
    else:
        # only return score
        return 0.5*np.sum(err**2)


def score_mia():
    """
    Multiple independent attributes.
    """
    pass


def loss_zero_one(z, targets):
    """
    """
    return np.sum(z==targets)
    
