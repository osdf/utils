"""

"""


import numpy as np
import scipy.linalg as la


def logsumexp(array, axis):
    """
    Compute sum of logs in a stable way.
    """
    pass


def score(weights, inputs, targets):
    """
    Compute score for _weights_, given
    _inputs_ and _targets_, the provided 
    supervision, need to be in 1-of-K coding.
    FIXME: weight decay???
    """
    _, di = inputs.shape
    _, dt = targets.shape
    y = np.dot(inputs, weights[:di*dt].reshape(di, dt))
    return np.sum(targets*(np.log(y) - logsumexp(y, axis=1)))


def predict(weights, inputs):
    """
    Predict target distribution 
    for _inputs_, given _weights_.
    """
    pass
