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


def test_cifar(cifar, opt, hdim, epochs=10, btsz=100,
        lr=1e-8, beta=0.9, eta0=2e-6, mu=0.02, lmbd=0.99,
        w=None):
    """
    Train on CIFAR-10 dataset. The data must be provided via _cifar_.
    If training should continue on some evolved weights, pass in _w_.
    """
    from opt import msgd, smd
    from misc import sigmoid
    from losses import ssd
    import pca
    #
    cifar -= np.atleast_2d(cifar.mean(axis=1)).T
    cifar /= np.atleast_2d(cifar.std(axis=1)).T

    cifar, comp, s = pca.pca(cifar, covered=0.99, whiten=True)

    n, idim = cifar.shape

    if w is None:
        if opt is smd:
            # needs np.complex initialization
            weights = np.zeros((idim*hdim + idim + hdim), dtype=np.complex)
            weights[:idim*hdim] = 1e-2 * np.random.randn(idim*hdim).ravel()
        else:
            weights = np.zeros((idim*hdim + idim + hdim))
            weights[:idim*hdim] = 1e-2 * np.random.randn(idim*hdim)
    else:
        print "Continue with provided weights w."
        weights = w

    structure = dict()
    structure["af"] = sigmoid
    structure["score"] = ssd
    structure["hdim"] = hdim
    
    print "Training begins ..."
    
    params = dict()
    params["x0"] = weights

    if opt is msgd or opt is smd:
        params["fandprime"] = score_grad
        params["nos"] = cifar.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": cifar}
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
        params["fprime"] = grad
        params["args"] = (structure, cifar)
        params["maxfun"] = epochs
        params["m"] = 50
        params["factr"] = 10.

    weights = opt(**params)[0]
    print "Training done."

    return weights
