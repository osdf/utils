"""
Tied AutoEncoder with one hidden layer.
"""


import numpy as np
import scipy.linalg as la


from misc import Dtable


def score(weights, structure, inputs, predict=False, 
        error=False, **params):
    """
    """
    hdim = structure["hdim"]
    A = structure["af"]
    _, idim = inputs.shape
    ih = idim * hdim
    hddn = A(np.dot(inputs, weights[:ih].reshape(idim, hdim)) + weights[ih:ih+hdim])
    if error:
        structure["hiddens"] = hddn
    z = np.dot(hddn, weights[:ih].reshape(idim, hdim).T) + weights[ih+hdim:]
    sc = structure["score"]
    return sc(z, inputs, predict=predict, error=error)


def score_grad(weights, structure, inputs, **params):
    """
    """
    hdim = structure["hdim"]
    _, idim = inputs.shape
    ih = idim * hdim
    af = structure["af"]
    g = np.zeros(weights.shape, dtype=weights.dtype)
    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, structure=structure, inputs=inputs, predict=False, error=True, **params)
    # recover saved hidden values
    hddn = structure["hiddens"]
    # weights are tied
    g[:ih] = np.dot(delta.T, hddn).ravel()
    g[ih+hdim:] = delta.sum(axis=0)
    dsc_dha = np.dot(delta, weights[:ih].reshape((idim, hdim))) * Dtable[af](hddn)
    g[:ih] += np.dot(inputs.T, dsc_dha).ravel()
    g[ih:ih+hdim] = dsc_dha.sum(axis=0)
    # clean up structure
    del structure["hiddens"]
    return sc, g


def grad(weights, structure, inputs, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, **params)
    return g


def predict(weights, structure, inputs, **params):
    """
    """
    return score(weights, structure, inputs, predict=True)



def check_the_grad(nos=1, idim=30, hdim=10, eps=1e-8, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    from misc import sigmoid
    from losses import ssd
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.randn(nos, idim)
    structure = dict()
    structure["hdim"] = hdim
    structure["af"] = sigmoid
    structure["score"] = ssd
    
    weights = np.zeros(idim*hdim + hdim + idim)
    weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim)
    
    args = dict()
    args["inputs"] = ins
    args["structure"] = structure
    #
    delta = check_grad(score, grad, weights, args, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[nn.py] check_the_grad FAILED. Delta is %f" % delta
    return True

