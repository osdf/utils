"""
Standard neural networks
"""


import numpy as np
import scipy.linalg as la


from misc import logsumexp


def score_xe(z, targets, predict=False, error=False):
    """
    """
    if predict:
        return np.argmax(z, axis=1)
    #
    xe = z - logsumexp(z, axis=1)
    if error:
        # score + error
        return xe, z - targets
    else:
        return xe


def score_ssd(z, targets, predict=False, error=False):
    """
    """
    if predict:
        return z
    error = z - targets 
    if error:
        # score + error
        return np.sum(error**2), error
    else:
        # only return score
        return np.sum(error**2)


def score_mia():
    """
    Multiple independent attributes.
    """
    pass


def fward(weights, structure, inputs, targets, predict=False, error=False):
    """
    structure: dictrionary with
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
        if error:
            pass
        idx = idy+l[1]
    # last layer: special care
    l = layers[:-1]
    idy = idx + l[0]*l[1]
    # no activation function, 
    # everyting will be handeled by score
    z = np.dot(z, weights[idx:idy].reshape(l[0], l[1])) \
                + weights[idy:idy+l[1]])
    return score(z, targets, predict, error)


def grad(weights, structure, inputs, targets):
    """
    """
    g = np.zeros(weights.shape)
    layers = structure["layers"]
    activs = structure["activs"]
    # forward pass through model,
    # need 'error' signal at the end.
    score, delta = fward(weights, structure, inputs, targets, 
            predict=False, error=True)
    # backprop through last layer, activation fct. is id.
    last layer deriv = np.dot((somethingfrom strucutre).T, delta)
    last layer biasderiv = delta.sum(axis=0)
    # new delta for one layer below
    delta = np.dot(delta, weights[....].reshape(....).T)
    for l, a in reversed(zip(layers[:-1], activs):
        dE_dA = delta * Dtable[a](somethingfromstructure)
        g[...] = np.dot(storeddatathatwasinputforl, dE_dA)
        g[biaseshere] = dE_dA.sum(axis=0)
        # backprop delta
        delta = np.dot(delta, weights[...].reshape(...).T)
        adapt some things, tracking parameters
    return g


def predict(weights, structure, inputs):
    """
    """
    return fward(weights, structure, inputs, 
            targets=None, predict=True)


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
