"""
Standard neural networks
"""


import numpy as np
import scipy.linalg as la


def score_sm():
    """
    """
    pass


def score_ssd():
    """
    """
    pass


def score_mia():
    """
    Multiple independent attributes.
    """
    pass


def fward(weights, structure, inputs, targets):
    """
    strcture: dictrionary with
    layers - list of pairs
    activations functions - list of act. func., apart from last layer - id
    score - list? 
    """
    layers = structure["layers"]
    activs = structure["activs"]
    score = structure["score"]
    z = inputs
    idx = 0
    for l, a in zip(layers[:-1], activs):
        idy = idx + l[0]*l[1]
        z = a(np.dot(z, weights[idx:idy].reshape(l[0], l[1])) \
                + weights[idy:idy+l[1]])
        idx = idy+l[1]
    l = layers[:-1]
    idy = idx + l[0]*l[1]
    z = np.dot(z, weights[idx:idy].reshape(l[0], l[1])) \
                + weights[idy:idy+l[1]])
    return score(z, targets)


def sigmoid(x):
    return (1 + np.tanh(activ/2.))/2.


def demo(hiddens, epochs, lr, btsz, lmbd):
    """
    """
    trainset, valset, testset = load_mnist()
    inputs, targets = trainset
    di = inputs.shape[1]
    dt = np.max(targets) + 1
    structure = {}
    structure["layers"] = [(di, hiddens), (hiddens, dt)]}
    structure["activs"] = [sigmoid]
    strucutre["score"] = score_sm
