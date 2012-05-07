"""
Contractive AutoEncoder with sigmoidal hidden layer.
"""


import numpy as np
import scipy.linalg as la


from misc import sigmoid, Dsigmoid
from losses import bKL


def score(weights, structure, inputs, predict=False, 
        error=False, **params):
    """
    Computes the sparisty penalty using exponential weighting.
    """
    hdim = structure["hdim"]
    _, idim = inputs.shape
    ih = idim * hdim

    hddn = sigmoid(np.dot(inputs, weights[:ih].reshape(idim, hdim)) + weights[ih:ih+hdim])
    z = np.dot(hddn, weights[:ih].reshape(idim, hdim).T) + weights[ih+hdim:]

    w = weights[:ih].reshape(idim, hdim)
    cae = np.sum(np.mean(Dsigmoid(hddn)**2, axis=0) * np.sum(w**2, axis=0))
    cae_weight = structure["cae"]
    cae *= cae_weight

    if error:
        structure["hiddens"] = hddn
    return structure["score"](z, inputs, predict=predict, error=error, addon=cae)


def score_grad(weights, structure, inputs, score=score, **params):
    """
    """
    hdim = structure["hdim"]
    m, idim = inputs.shape
    ih = idim * hdim
    g = np.zeros(weights.shape, dtype=weights.dtype)
    
    # forward pass through model,
    # need 'error' signal at the end.
    sc, delta = score(weights, structure=structure, inputs=inputs, predict=False, error=True, **params)
    
    # recover saved hidden values
    hddn = structure["hiddens"]
    
    # weights are tied
    g[:ih] = np.dot(delta.T, hddn).ravel()
    g[ih+hdim:] = delta.sum(axis=0)
    
    # derivative of cae cost
    w = weights[:ih].reshape(idim, hdim)
    cae_grad = np.mean(Dsigmoid(hddn)**2, axis=0) * w
    cae_grad += (np.dot(inputs.T, (Dsigmoid(hddn)**2 * (1-2*hddn)))/m * np.sum(w**2, axis=0))
    cae_weight = structure["cae"]
    g[:ih] += cae_weight * 2 * cae_grad.ravel()

    dsc_dha = Dsigmoid(hddn) * np.dot(delta, weights[:ih].reshape(idim, hdim))
    g[:ih] += np.dot(inputs.T, dsc_dha).ravel()

    g[ih:ih+hdim] = dsc_dha.sum(axis=0)
    # clean up structure
    del structure["hiddens"]
    return sc, g


def grad(weights, structure, inputs, score=score, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, score=score, **params)
    return g


def check_the_grad(nos=100, idim=30, hdim=10, eps=1e-8, verbose=False):
    """
    Check gradient computation for Neural Networks.
    """
    #
    from opt import check_grad
    from misc import sigmoid
    from losses import ssd, mia
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.randn(nos, idim)
    structure = dict()
    structure["hdim"] = hdim
    structure["score"] = mia 
    structure["cae"] = 3 
    
    weights = np.zeros(idim*hdim + hdim + idim)
    weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim)
    
    args = dict()
    args["inputs"] = ins
    args["structure"] = structure
    #
    delta = check_grad(score, grad, weights, args, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[cae.py] check_the_grad FAILED. Delta is %f" % delta
    return True
