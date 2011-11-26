"""

"""


import numpy as np
import scipy.linalg as la


def logsumexp(array, axis):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way.
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


def sigmoid(x):
    return (1 + np.tanh(activ/2.))/2.


def one2K(classes):
    """
    Given a numeric encoding of
    class membership in _classes_, 
    produce a 1-of-K encoding.
    """
    c = classes.max() + 1
    targets = np.zeros((len(classes), c))
    targets[classes] = 1
    return targets


def K2one(onehotcoding):
    return np.argmax(onehotcoding, axis=1)
