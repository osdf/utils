"""
ICA -- Independent Component Analysis

This is based on 
ICA with Reconstruction Cost for Efficient
Overcomplete Feature Learning, by Quoc V. Le et al.

If you are looking for 'standard' ICA
you need to go somewhere else, e.g.
scikits.learn.
"""


import numpy as np
import scipy as sp


from losses import ssd
from misc import Dtable


def score(weights, structure, inputs,
        predict=False, error=False):
    """
    """
    n, di = inputs.shape
    dh = weights.shape[0]/di
    # independent components (ic): linear projections
    ic = np.dot(inputs, weights.reshape(di, dh))
    if predict:
        return ic 

    z = np.dot(ic, weights.reshape(di, dh).T)
    # smooth l1 penalty cost
    l1 = structure["l1"]
    pl1 = structure["lmbd"] * np.sum(l1(ic))
    if error:
        sc, err = ssd(z, inputs, predict=False, error=True)
        sc += pl1
        # returns also first derivative of rec. error!
        return sc, err 
    else:
        sc = ssd(z, inputs, predict=False, error=False) + structure["lmbd"]*pl1
        return sc


def score_grad(weights, structure, inputs, **params):
    """
    """
    # get _compelete_ score and first deriv of rec. error
    sc, delta = score(weights, inputs=inputs, structure=structure, 
            predict=False, error=True, **params)
    n, di = inputs.shape
    dh = weights.shape[0]/di
    l1 = structure["l1"]

    ic = np.dot(inputs, weights.reshape(di, dh))
    Dsc_Dic = np.dot(delta, weights.reshape(di, dh))

    g = np.zeros(weights.shape, dtype=weights.dtype)
    g += np.dot(ic.T, delta).T.ravel()
    g += np.dot(inputs.T, Dsc_Dic).ravel()
    g += structure["lmbd"] * np.dot(inputs.T, Dtable[l1](ic)).ravel()
    return sc, g


def grad(weights, structure, inputs, **params):
    """
    """
    _, g = score_grad(weights, structure, inputs, **params)
    return g


def score_grad_norm(weights, structure, inputs, eps=1e-8, **params):
    """
    """
    n, di = inputs.shape
    dh = weights.shape[0]/di

    w = weights.reshape(ind, hid)
    _l2 = sp.sqrt(np.sum(w**2, axis=0) + eps)
    _w = w/_l2

    # gradient from ball projection + reshaped
    sc, _g = score_grad(_w.ravel(), structure, inputs, **params)
    _g = _g.reshape(ind, hid)

    g = _g/_l2 - _w * (np.sum(_g * w, axis=0))/(_l2 **2)
    return sc, g.ravel()


def grad_norm(weights, structure, inputs, eps=10**-5, **params):
    """
    """
    _, g = score_grad_norm(weights, structure, inputs, eps=10**-5, **params)
    return g


def check_the_grad(nos=1, ind=30, outd=10, eps=1e-8, verbose=False):
    """
    Check gradient computation.
    """
    from opt import check_grad
    from misc import logcosh, sqrtsqr
    # number of input samples (nos)
    # with dimension ind each
    ins = np.random.randn(nos, ind)
    
    weights = 0.001 * np.random.randn(ind, outd).ravel()

    structure = dict()
    structure["l1"] = sqrtsqr
    structure["lmbd"] = 1 

    args = dict()
    args["inputs"] = ins
    args["structure"] = structure

    delta = check_grad(score, grad, weights, args, eps=eps, verbose=verbose)
    
    assert delta < 1e-4, "[ica.py] check_the_grad FAILED. Delta is %f" % delta
    return True


def test_cifar(gray, opt, dh, lambd, epochs=10, btsz=100,
        lr=1e-8, beta=0.9, eta0=2e-6, mu=0.02, lmbd=0.99,
        w=None):
    """
    Train on CIFAR-10 dataset. The data must be provided via _gray_. 
    The optimizer is in _opt_ -- several of the following parameters 
    are related to _opt_. The resulting dimension per sample is _dh_.
    _lambd_ is the weighting of the ICA objective, see comments at the
    top of this file. _epochs_ is the number of passes over the training
    set. If stochastic descent is used, _btsz_ refers to the size of
    a minibatch, _lr_ to the learning rate and _beta_ to the momentum
    factor. Note that _lr_ needs to be very small in order to get a decent
    training result. _eta0_, _mu_ and _lmbd_ are parameters for
    stochastic meta descent (SMD), see opt.smd for more explanation.
    If training should continue on some evolved weights, pass in _w_.
    """
    from opt import msgd, smd
    from misc import logcosh, sqrtsqr
    #
    n, di = gray.shape
    if w is None:
        if opt is smd:
            # needs np.complex initialization
            weights = np.zeros((ind*outd), dtype=np.complex)
            weights[:] = 0.001 * np.random.randn(di, dh).ravel()
        else:
            weights = 0.001 * np.random.randn(di, dh).ravel()
    else:
        print "Continue with provided weights w."
        weights = w
        #
    structure = dict()
    structure["l1"] = sqrtsqr
    # parameter _lmbd_ is for SMD!
    structure["lmbd"] = lambd
    #
    print "Training starts ..."
    params = dict()
    params["x0"] = weights

    if opt is msgd or opt is smd:
        params["fandprime"] = score_grad_norm
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
        params["fprime"] = grad_norm
        params["args"] = (structure, gray)
        params["maxfun"] = epochs
        params["m"] = 50
        params["factr"] = 10.

    weights = opt(**params)[0]
    print "Training done."

    return weights
