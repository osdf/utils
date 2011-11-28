"""

"""


import numpy as np
import scipy.linalg as la


def sigmoid(x):
    return (1 + np.tanh(x/2.))/2.


def Dsigmoid(y):
    """
    Given y = sigmoid(x),
    what is dy/dx in terms of
    y.
    """
    return y*(1-y)


def Dtanh(y):
    """
    Given y = tanh(x),
    what is dy/dx in terms of
    y.
    """
    return 1 - y**2


Dtable = {
        sigmoid: Dsigmoid,
        np.tanh: Dtanh
        }


def logsumexp(array, axis):
    """
    Compute log of (sum of exps) 
    along _axis_ in _array_ in a 
    stable way.
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


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


def load_mnist():
    """
    Get standard MNIST: training, validation, testing.
    File 'mnist.pkl.gz' must be in path.
    Download it from ...
    """
    import gzip, cPickle
    print "Reading mnist.pkl.gz ..."
    f = gzip.open('mnist.pkl.gz','rb')
    trainset, valset, testset = cPickle.load(f)
    return trainset, valset, testset
