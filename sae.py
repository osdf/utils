"""
Sparse (tied) AutoEncoder with one hidden layer.
This implementation only makes sense with hidden bernoulli
variables.
"""


import numpy as np
import scipy.linalg as la


from misc import Dtable
from losses import bKL


def true_score(weights, structure, inputs, predict=False, 
        error=False, **params):
    """
    Computes the sparsity penalty according to 'correct' formula,
    but needs a full pass over training set. Use for numerical gradient
    check.
    """
    hdim = structure["hdim"]
    A = structure["af"]
    _, idim = inputs.shape
    ih = idim * hdim

    hddn = A(np.dot(inputs, weights[:ih].reshape(idim, hdim)) + weights[ih:ih+hdim])
    z = np.dot(hddn, weights[:ih].reshape(idim, hdim).T) + weights[ih+hdim:]
    
    # sparsity penalty is bernoulli KL
    rho_hat = hddn.mean(axis=0)
    sparse_pen = structure["beta"] * np.sum(bKL(structure["rho"], rho_hat)) 

    if error:
        structure["hiddens"] = hddn
        structure["rho_hat"] = rho_hat
    
    return structure["score"](z, inputs, predict=predict, error=error, addon=sparse_pen)


def score(weights, structure, inputs, predict=False, 
        error=False, **params):
    """
    Computes the sparisty penalty using exponential weighting.
    """
    hdim = structure["hdim"]
    A = structure["af"]
    _, idim = inputs.shape
    ih = idim * hdim
    rho_hat = structure["rho_hat"]
    # exponential decay for rho_hat over minibatches
    lmbd = structure["lmbd"]

    hddn = A(np.dot(inputs, weights[:ih].reshape(idim, hdim)) + weights[ih:ih+hdim])
    z = np.dot(hddn, weights[:ih].reshape(idim, hdim).T) + weights[ih+hdim:]
    
    # sparsity penalty is bernoulli KL
    # avoid full passes over dataset via exponential decay
    rho_hat *= lmbd
    rho_hat += (1 - lmbd) * hddn.mean(axis=0)
    sparse_pen = structure["beta"] * np.sum(bKL(structure["rho"], rho_hat)) 
    
    if error:
        structure["hiddens"] = hddn
    return structure["score"](z, inputs, predict=predict, error=error, addon=sparse_pen)


def score_grad(weights, structure, inputs, score=true_score, **params):
    """
    """
    hdim = structure["hdim"]
    m, idim = inputs.shape
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
    # derivative of sparsity wrt to ha
    dsparse_dha = -structure["rho"]/structure["rho_hat"] + (1-structure["rho"])/(1-structure["rho_hat"])
    dsc_dha = Dtable[af](hddn) * (np.dot(delta, weights[:ih].reshape((idim, hdim))) + structure["beta"]*dsparse_dha/m)
    g[:ih] += np.dot(inputs.T, dsc_dha).ravel()
    g[ih:ih+hdim] = dsc_dha.sum(axis=0)
    # clean up structure
    del structure["hiddens"]
    return sc, g


def grad(weights, structure, inputs, score=true_score, **params):
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
    structure["af"] = sigmoid
    structure["score"] = ssd
    structure["beta"] = 0.7
    structure["rho"] = 0.01
    
    weights = np.zeros(idim*hdim + hdim + idim)
    weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim)
    
    args = dict()
    args["inputs"] = ins
    args["structure"] = structure
    #
    delta = check_grad(true_score, grad, weights, args, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[sae.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def test_cifar(gray, opt, hdim, epochs=10, btsz=100,
        lr=1e-8, beta=0.9, eta0=2e-6, mu=0.02, lmbd=0.99,
        w=None):
    """
    Train on CIFAR-10 dataset. The data must be provided via _gray_. 
    If training should continue on some evolved weights, pass in _w_.
    """
    from functools import partial
    from opt import msgd, smd
    from misc import sigmoid
    from losses import ssd
    #
    n, idim = gray.shape
    if w is None:
        if opt is smd:
            # needs np.complex initialization
            weights = np.zeros((idim*hdim + idim + hdim), dtype=np.complex)
            weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim).ravel()
        else:
            weights = np.zeros((idim*hdim + idim + hdim))
            weights[:idim*hdim] = 0.001 * np.random.randn(idim*hdim)
    else:
        print "Continue with provided weights w."
        weights = w

    structure = dict()
    structure["af"] = sigmoid
    structure["score"] = ssd
    structure["hdim"] = hdim
    structure["beta"] = 0.7
    structure["rho"] = 0.01
    structure["rho_hat"] = np.zeros(hdim)
    structure["lmbd"] = 0.9

    print "Training begins ..."

    params = dict()
    params["x0"] = weights

    if opt is msgd or opt is smd:
        params["fandprime"] = partial(score_grad, score=score)
        params["nos"] = gray.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": gray}
        params["epochs"] = epochs
        params["btsz"] = btsz
        # msgd
        params["lr"] = lr 
        params["beta"] = beta
        # smd
        params["eta0"] = eta0
        params["mu"] = mu
        params["lmbd"] = lmbd

        params["verbose"] = True
    else:
        # opt from scipy
        params["func"] = score
        params["fprime"] = partial(grad, score=score)
        params["args"] = (structure, gray)
        params["maxfun"] = epochs
        params["m"] = 50
        params["factr"] = 10.

    weights = opt(**params)[0]
    print "Training done."

    return weights
