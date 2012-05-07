"""
Sparse Filtering.

Sparse filtering, J. Ngiam, P. Koh, Z. Chen, S. Bhaskar, A.Y. Ng.
NIPS 2011.

This implementation does not optimize
the _sparse filtering objective_ from
the paper (eq. 1) -- the l1 penalty
is dropped. It uses a nonlinear single layer
network that produces the feature matrix. 
"""


import numpy as np


from misc import Dtable


def score(weights, structure, inputs, gradient=False, **params):
    """
    _inputs_ are per row and should be
    normalized (mean==0 per example).
    """
    n, di = inputs.shape
    af = structure["af"]
    eps = structure["eps"]
    dz = weights.shape[0]/di
    
    z = np.dot(inputs, weights.reshape(di, dz))
    f = af(z)
    
    # L2 Normalize every feature (==columns) over inputs
    col_l2 = np.sqrt(np.sum(f**2, axis=0) + eps)
    ftilde = f/col_l2
    
    # L2 normalize every input
    row_l2 = np.sqrt(np.sum(ftilde**2, axis=1) + eps)
    row_l2.resize(n,1)
    fhat = ftilde/row_l2

    # save for backpropagation pass
    if gradient:
        structure["z"] = z
        structure["ftilde"] = ftilde
        structure["fhat"] = fhat
        structure["row_l2"] = row_l2
        structure["col_l2"] = col_l2
    return np.sum(fhat)


def score_grad(weights, structure, inputs, **params):
    """
    """
    sc = score(weights, structure, inputs, gradient=True, **params)
    
    n, di = inputs.shape
    af = structure["af"]
    z = structure["z"]
    fhat = structure["fhat"]
    ftilde = structure["ftilde"]
    row_l2 = structure["row_l2"]
    col_l2 = structure["col_l2"]

    rowsum = np.sum(fhat, axis=1)
    rowsum.resize(n, 1)
    dfhat_dftilde = (1 - fhat * rowsum)/row_l2
    dfhat_df = np.sum(ftilde * dfhat_dftilde, axis=0)
    dfhat_dz = Dtable[af](z) * (dfhat_dftilde - ftilde*dfhat_df)/col_l2
    grad = np.dot(inputs.T, dfhat_dz).ravel()

    del structure["fhat"]
    del structure["ftilde"]
    del structure["row_l2"]
    del structure["col_l2"]
    del structure["z"]

    return sc, grad


def grad(weights, structure, inputs, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, **params)
    return g


def check_the_grad(nos=100, ind=30, outd=10, eps=1e-8, verbose=False):
    """
    Check gradient computation.

    _nos_: number of samples
    _ind_: dimension of one sample
    _outd_: number of filters
    """
    from opt import check_grad
    from misc import sqrtsqr

    ins = np.random.randn(nos, ind)
    weights = 0.001 * np.random.randn(ind, outd).ravel()

    structure = dict()
    structure["af"] = sqrtsqr
    structure["eps"] = 1e-8

    args = dict()
    args["inputs"] = ins
    args["structure"] = structure

    delta = check_grad(score, grad, weights, args, eps=eps, verbose=verbose)
    assert delta < 1e-4, "[sf.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def test_cifar(gray, opt, outd, epochs=10, btsz=100,
        lr=1e-8, beta=0.9, w=None):
    """
    """
    from opt import msgd
    from misc import sqrtsqr
    #
    n, ind = gray.shape

    if w is None:
        weights = 0.001 * np.random.randn(ind, outd).ravel()
    else:
        print "Continue with provided weights w."
        weights = w

    structure = dict()
    structure["af"] = sqrtsqr
    structure["eps"] = 1e-8

    print "Training starts ..."

    params = dict()
    params["x0"] = weights
    if opt is msgd:
        params["fandprime"] = score_grad
        params["nos"] = gray.shape[0]
        params["args"] = {"structure": structure}
        params["batch_args"] = {"inputs": gray}
        params["epochs"] = epochs
        params["btsz"] = btsz
        params["lr"] = lr 
        params["beta"] = beta
        params["verbose"] = True
    else:
        # opt from scipy
        params["func"] = score
        params["fprime"] = grad
        params["args"] = (structure, gray)
        params["maxfun"] = epochs
        params["m"] = 20
        params["factr"] = 10.

    weights = opt(**params)[0]
    print "Training done."

    return weights
